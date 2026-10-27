// Public Metal API harness only. No private APIs, product modules, or guest data.
import Foundation
import Metal
import CryptoKit
import Darwin

struct Input: Decodable {
    let id: String
    let x: Float
    let lower: Float
    let upper: Float
}
struct Failure: Error { let message: String }
func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw Failure(message: message) }
}
func digest(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}
func hex(_ value: UInt32, _ half: Bool = false) -> String {
    String(format: half ? "0x%04x" : "0x%08x", value)
}
func data<T>(_ values: [T]) -> Data { values.withUnsafeBytes { Data($0) } }

func run() throws -> [String: Any] {
    try require(CommandLine.arguments.count == 3, "usage: probe shader.metal cases.json")
    let sourceData = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
    let caseData = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[2]))
    guard let source = String(data: sourceData, encoding: .utf8) else {
        throw Failure(message: "shader is not UTF-8")
    }
    let cases = try JSONDecoder().decode([Input].self, from: caseData)
    try require(!cases.isEmpty && cases.count <= 128, "case count outside bound")
    try require(Set(cases.map { $0.id }).count == cases.count, "duplicate case id")
    for c in cases {
        for v in [c.x, c.lower, c.upper] {
            try require(v.isFinite && Float(Float16(v)) == v, "input not finite/exact in both domains")
        }
    }
    try require(MemoryLayout<SIMD4<Float>>.stride == 16, "float4 ABI")
    try require(MemoryLayout<SIMD4<UInt16>>.stride == 8, "half4 ABI")
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw Failure(message: "GPU_UNAVAILABLE: MTLCreateSystemDefaultDevice returned nil")
    }
    try require(device.name.hasPrefix("Apple"), "not an Apple Metal device: \(device.name)")
    guard let queue = device.makeCommandQueue() else { throw Failure(message: "command queue unavailable") }

    func buffer(_ bytes: Data) throws -> MTLBuffer {
        try bytes.withUnsafeBytes { raw in
            guard let address = raw.baseAddress,
                  let b = device.makeBuffer(bytes: address, length: raw.count, options: .storageModeShared)
            else { throw Failure(message: "buffer creation failed") }
            return b
        }
    }
    func words(_ b: MTLBuffer, _ count: Int) -> [UInt32] {
        Array(UnsafeBufferPointer(start: b.contents().bindMemory(to: UInt32.self, capacity: count), count: count))
    }

    var observations: [[String: Any]] = []
    var checks: [String] = []
    var compileRows: [[String: Any]] = []
    for fast in [false, true] {
        let options = MTLCompileOptions()
        options.languageVersion = .version3_1
        options.fastMathEnabled = fast
        let library = try device.makeLibrary(source: source, options: options)
        compileRows.append([
            "fastMathEnabled": fast, "languageVersion": "Metal 3.1",
            "method": "MTLDevice.makeLibrary(source:options:)",
            "functions": library.functionNames.sorted()
        ])
        for half in [false, true] {
            let domain = half ? "f16" : "f32"
            let names = half ? ["half_clamp", "half_fmin_fmax"] :
                ["fast_clamp", "precise_clamp", "fast_fmin_fmax", "precise_fmin_fmax"]
            guard let function = library.makeFunction(name: "probe_" + domain) else {
                throw Failure(message: "missing function for " + domain)
            }
            let pipeline = try device.makeComputePipelineState(function: function)
            // Reuse the very same compiled pipeline with two runtime buffer orders.
            for reverse in [false, true] {
                let batch = reverse ? Array(cases.reversed()) : cases
                let n = batch.count
                let inputs: Data = half ? data(batch.map {
                    SIMD4<UInt16>(Float16($0.x).bitPattern, Float16($0.lower).bitPattern,
                                  Float16($0.upper).bitPattern, 0)
                }) : data(batch.map { SIMD4<Float>($0.x, $0.lower, $0.upper, 0) })
                let input = try buffer(inputs)
                let echo = try buffer(data([UInt32](repeating: 0xdeadbeef, count: n * 4)))
                let raw = try buffer(data([UInt32](repeating: 0xdeadbeef, count: n * 4)))
                let td = MTLTextureDescriptor.texture2DDescriptor(
                    pixelFormat: .rgba8Unorm, width: n, height: names.count, mipmapped: false)
                td.storageMode = .shared
                td.usage = [.shaderWrite, .shaderRead]
                guard let texture = device.makeTexture(descriptor: td),
                      let command = queue.makeCommandBuffer(),
                      let encoder = command.makeComputeCommandEncoder() else {
                    throw Failure(message: "texture/command/encoder allocation failed")
                }
                let initial = [UInt8](repeating: 0x5a, count: n * names.count * 4)
                initial.withUnsafeBytes {
                    texture.replace(region: MTLRegionMake2D(0, 0, n, names.count), mipmapLevel: 0,
                                    withBytes: $0.baseAddress!, bytesPerRow: n * 4)
                }
                encoder.setComputePipelineState(pipeline)
                encoder.setBuffer(input, offset: 0, index: 0)
                encoder.setBuffer(echo, offset: 0, index: 1)
                encoder.setBuffer(raw, offset: 0, index: 2)
                encoder.setTexture(texture, index: 0)
                encoder.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                encoder.endEncoding()
                command.commit()
                let deadline = Date().addingTimeInterval(30)
                while command.status != .completed && command.status != .error && Date() < deadline {
                    Thread.sleep(forTimeInterval: 0.001)
                }
                try require(command.status == .completed,
                            "GPU error/30-second timeout: \(String(describing: command.error))")
                let echoed = words(echo, n * 4)
                let output = words(raw, n * 4)
                var pixels = [UInt8](repeating: 0, count: initial.count)
                pixels.withUnsafeMutableBytes {
                    texture.getBytes($0.baseAddress!, bytesPerRow: n * 4,
                                     from: MTLRegionMake2D(0, 0, n, names.count), mipmapLevel: 0)
                }
                for (slot, c) in batch.enumerated() {
                    let expectedEcho: [UInt32] = [c.x, c.lower, c.upper].map {
                        half ? UInt32(Float16($0).bitPattern) : $0.bitPattern
                    } + [UInt32(slot)]
                    if Array(echoed[slot * 4 ..< slot * 4 + 4]) != expectedEcho {
                        checks.append("runtime input echo mismatch: \(domain)/\(c.id)")
                    }
                    let expected = min(max(c.x, c.lower), c.upper)
                    let expectedBits = half ? UInt32(Float16(expected).bitPattern) : expected.bitPattern
                    let ordered = c.lower <= c.upper
                    let comparisonIndex = half ? 1 : 3
                    for op in names.indices {
                        let bits = output[slot * 4 + op]
                        let pixelOffset = (op * n + slot) * 4
                        let rgba = Array(pixels[pixelOffset ..< pixelOffset + 4])
                        let comparator = half ? op == 1 : op >= 2
                        // Reversed clamp is OBSERVATION ONLY: never an expected-value assertion.
                        if (ordered || comparator) && bits != expectedBits {
                            checks.append("defined-domain/control mismatch: \(domain)/\(c.id)/\(names[op])")
                        }
                        if rgba[0] != rgba[1] || rgba[1] != rgba[2] || rgba[3] != 255 {
                            checks.append("RGBA8 store shape mismatch: \(domain)/\(c.id)/\(names[op])")
                        }
                        var row: [String: Any] = [
                            "domain": domain, "case": c.id, "slot": slot,
                            "buffer_order": reverse ? "reverse" : "forward", "fastMathEnabled": fast,
                            "operation": names[op], "x": c.x, "lower": c.lower, "upper": c.upper,
                            "bounds": c.lower < c.upper ? "ordered" : (ordered ? "equal" : "reversed"),
                            "input_bits": expectedEcho.prefix(3).map { hex($0, half) },
                            "echo_bits": echoed[slot * 4 ..< slot * 4 + 3].map { hex($0, half) },
                            "output_bits": hex(bits, half), "rgba8": rgba,
                            "same_bits_as_explicit_comparator": bits == output[slot * 4 + comparisonIndex],
                            "runtime_input_buffer_sha256": digest(inputs),
                            "clamp_domain_defined": ordered
                        ]
                        if half {
                            let widened = output[slot * 4 + 2 + op]
                            row["gpu_widened_f32_bits"] = hex(widened)
                            let hostWidened = Float(Float16(bitPattern: UInt16(truncatingIfNeeded: bits)))
                            if hostWidened.isFinite && widened != hostWidened.bitPattern {
                                checks.append("half exact-widening mismatch: \(c.id)/\(names[op])")
                            }
                        }
                        observations.append(row)
                    }
                }
            }
        }
    }
    let identity: [String: Any] = [
        "name": device.name, "registry_id": String(device.registryID),
        "unified_memory": device.hasUnifiedMemory, "low_power": device.isLowPower,
        "removable": device.isRemovable
    ]
    return [
        "schema": "probe-v1", "native_execution": true,
        "device": identity,
        "os": ProcessInfo.processInfo.operatingSystemVersionString,
        "shader_sha256": digest(sourceData), "cases_sha256": digest(caseData),
        "compiles": compileRows, "observations": observations, "validation_failures": checks,
        "scope": "finite synthetic scalar inputs only; no nonfinite-input experiment",
        "limitation": "MSL reversed bounds are undefined; observations are not portable semantics. " +
                      "Explicit comparator expressions may be optimized to the same instructions. " +
                      "RGBA8 may mask distinct negative floats; raw output bits are authoritative. " +
                      "A paravirtual device is not bare-M1 GPU conformance."
    ]
}

do {
    let result = try run()
    FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: result,
                                                               options: [.prettyPrinted, .sortedKeys]))
    FileHandle.standardOutput.write(Data([10]))
    exit((result["validation_failures"] as? [String] ?? ["missing validation"]).isEmpty ? 0 : 1)
} catch {
    fputs("finite synthetic Metal probe failed: \(error)\n", stderr)
    exit(1)
}
