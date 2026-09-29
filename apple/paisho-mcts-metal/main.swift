import Foundation
import Metal

// Persistent framed service. No neural network or floating point GPU arithmetic.
// A command has a bounded lifetime; a failed GPU command is never a CPU result.
func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8)); exit(1)
}
func readExact(_ count: Int, allowEOF: Bool = false) -> Data? {
    var data = Data()
    while data.count < count {
        do {
            guard let part = try FileHandle.standardInput.read(upToCount: count-data.count), !part.isEmpty else {
                if data.isEmpty && allowEOF { return nil }
                fail("truncated PMG1 request")
            }
            data.append(part)
        } catch { fail("PMG1 input: \(error)") }
    }
    return data
}
let capacity = 32768
let args = CommandLine.arguments
if args.count != 2 { fail("usage: paisho-mcts-metal evaluate.metal") }
guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else { fail("Metal device unavailable") }
let pipeline: MTLComputePipelineState
do {
    let source = try String(contentsOfFile: args[1], encoding: .utf8)
    let library = try device.makeLibrary(source: source, options: nil)
    guard let function = library.makeFunction(name: "features") else { fail("missing features kernel") }
    pipeline = try device.makeComputePipelineState(function: function)
} catch { fail("Metal compilation: \(error)") }
guard let input = device.makeBuffer(length: capacity*289, options: .storageModeShared),
      let output = device.makeBuffer(length: capacity*16, options: .storageModeShared) else { fail("Metal allocation") }
FileHandle.standardOutput.write(Data("PMG1".utf8))
while let header = readExact(4, allowEOF: true) {
    let count = header.enumerated().reduce(0) { $0 | (Int($1.element) << (8*$1.offset)) }
    if count < 1 || count > capacity { fail("invalid PMG1 batch size") }
    let data = readExact(count*289)!
    data.withUnsafeBytes { input.contents().copyMemory(from: $0.baseAddress!, byteCount: data.count) }
    guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else { fail("Metal command allocation") }
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input, offset: 0, index: 0)
    encoder.setBuffer(output, offset: 0, index: 1)
    encoder.dispatchThreadgroups(MTLSize(width: count,height: 1,depth: 1), threadsPerThreadgroup: MTLSize(width: 32,height: 1,depth: 1))
    encoder.endEncoding()
    let done = DispatchSemaphore(value: 0)
    command.addCompletedHandler { _ in done.signal() }
    command.commit()
    if done.wait(timeout: .now()+10) == .timedOut { fail("Metal command exceeded 10 seconds") }
    if command.status != .completed { fail("Metal command failed: \(String(describing: command.error))") }
    FileHandle.standardOutput.write(Data(bytes: output.contents(), count: count*16))
}
