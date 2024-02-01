import Cocoa
import Foundation

final class Value {
    enum Operator: CustomStringConvertible {
        case add
        case multiply
        case tanh
        case exp
        case pow
        case substract
        
        var description: String {
            switch self {
            case .add:
                "+"
            case .multiply:
                "*"
            case .tanh:
                "tanh"
            case .exp:
                "exp"
            case .pow:
                "**"
            case .substract:
                "-"
            }
        }
    }
    
    var data: Double
    var label: String?
    var grad: Double
    let op: Operator?
    var backward: (() -> Void)?
    let children: Set<Value>
    
    init(_ data: Double,
         children: Set<Value> = [],
         label: String? = nil,
         op: Operator? = nil,
         grad: Double = 0.0,
         backward: (() -> Void)? = nil
    ) {
        self.data = data
        self.children = children
        self.label = label
        self.grad = grad
        self.op = op
        self.backward = backward
    }
    
}

extension Value: ExpressibleByFloatLiteral, ExpressibleByIntegerLiteral {
    convenience init(floatLiteral value: Double) {
        self.init(value)
    }
    
    convenience init(integerLiteral value: IntegerLiteralType) {
        self.init(Double(value))
    }
}

extension Value: CustomStringConvertible {
    var description: String {
        String(format: "%@ | op %@ | data %.4f | grad: %.4f", label ?? "", op?.description ?? "", data, grad)
    }
}

extension Value: Equatable, Hashable {
    static func == (lhs: Value, rhs: Value) -> Bool {
        lhs === rhs
    }
    
    func hash(into hasher: inout Hasher) {
          hasher.combine(objectIdentifier)
      }

    var objectIdentifier: ObjectIdentifier {
        return ObjectIdentifier(self)
    }
}

// expressions
extension Value {
    func exp() -> Value {
        let v = Darwin.exp(data)
        let out = Value(
            v,
            children: [self],
            op: .exp
        )
        
        out.backward = { [unowned self] in
            self.grad += out.data * out.grad
        }
        
        return out
    }
    
    func tanh() -> Value {
        let t = Darwin.tanh(data)
        let out = Value(
            t,
            children: [self],
            op: .tanh
        )
        
        out.backward = { [unowned self] in
            self.grad += (1 - Darwin.pow(t, 2)) * out.grad
        }
        
        return out
    }
    
    func pow(_ n: Double) -> Value {
        let out = Value(
            Darwin.pow(data, n),
            children: [self],
            op: .pow
        )
        
        out.backward = { [unowned self] in
            self.grad += n * Darwin.pow(data, n - 1)
        }
        
        return out
    }
    
    static func +(lhs: Value, rhs: Value) -> Value {
        let out = Value(
            lhs.data + rhs.data,
            children: [lhs, rhs],
            op: .add
        )
        
        out.backward = {
            lhs.grad += out.grad
            rhs.grad += out.grad
        }
        
        return out
    }
    
    static func -(lhs: Value, rhs: Value) -> Value {
        lhs + (rhs * -1)
    }
    
    static func *(lhs: Value, rhs: Value) -> Value {
        let out = Value(
            lhs.data * rhs.data,
            children: [lhs, rhs],
            op: .multiply
        )
        
        out.backward = {
            lhs.grad += out.grad * rhs.data
            rhs.grad += out.grad * lhs.data
        }
        
        return out
    }
    
    static func /(lhs: Value, rhs: Value) -> Value {
        // division is just a special case of exponential
        // a / b = a * (1/b) = 1 * (b**-1)
        lhs * rhs.pow(-1)
    }
}

func backward(_ v: Value) {
    var topo = [Value]()
    var visited = Set<Value>()
    
    func buildTopo(_ n: Value) {
        guard !visited.contains(n) else {
            return
        }
        
        visited.insert(n)
        n.children.forEach { child in
            buildTopo(child)
        }
        topo.append(n)
    }
    
    buildTopo(v)
    
    v.grad = 1
    for v in topo.reversed() {
        v.backward?()
    }
}

let x1 = Value(2.0, label: "x1")
let x2 = Value(0.0, label: "x2")


let w1 = Value(-3.0, label: "w1")
let w2 = Value(1.0, label: "w2")

let b = Value(6.88137358, label: "b")

let x1w1 = x1 * w1
x1w1.label = "x1 * w1"
let x2w2 = x2 * w2
x2w2.label = "x2 * w2"

let x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = "x1*w1 + x2*w2"


let n = x1w1x2w2 + b
n.label = "n"

//var out = n.tanh()
var e = (2 * n).exp(); e.label = "e"
var out = (e - 1) / (e + 1) // broken down of tanh
out.label = "o"


backward(out)

final class Neuron {
    let weights: [Value]
    let bias: Value
    
    var parameters: [Value] {
        weights + [bias]
    }
    
    init(numberOfInputs: Int) {
        self.weights = (0..<numberOfInputs).map { _ in Value(Double.random(in: -1...1)) }
        self.bias = Value(Double.random(in: -1...1))
    }
    
    func callAsFunction(_ x: [Value]) -> Value {
        assert(x.count == weights.count, "inputs and weights count mismatch")
        
        let activation = zip(x, weights).reduce(bias, {
            $0 + $1.0 * $1.1
        })
        
        let out = activation.tanh()
        return out
    }
}

final class Layer {
    let neurons: [Neuron]
    
    var parameters: [Value] {
        neurons.flatMap(\.parameters)
    }
    
    init(numberOfInputs: Int, numberOfOutputs: Int) {
        self.neurons = (0..<numberOfOutputs).map { _ in
            Neuron(numberOfInputs: numberOfInputs)
        }
    }
    
    func callAsFunction(_ x: [Value]) -> [Value] {
        neurons.map { $0(x) }
    }
}

final class MLP {
    let layers: [Layer]
    
    var parameters: [Value] {
        layers.flatMap(\.parameters)
    }

    init(numberOfInputs: Int, numberOfOutputs: [Int]) {
        let allLayers = [numberOfInputs] + numberOfOutputs
        self.layers = (0..<numberOfOutputs.count).map {
            Layer(numberOfInputs: allLayers[$0], numberOfOutputs: allLayers[$0+1])
        }
    }
    
    func callAsFunction(_ x: [Value]) -> [Value] {
        var x = x
        layers.forEach {
            x = $0(x)
        }
        return x
    }
}

let mlp = MLP(numberOfInputs: 3, numberOfOutputs: [4, 4, 1])

let xs: [[Value]] = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
let ys: [Value] = [1.0, -1.0, -1.0, 1.0]

var loss: Value
for i in (0..<5) {
    print("__ iteration \(i) __")
    let yPred = xs.map { mlp($0)[0] }
    
    // forward pass
    loss = zip(ys, yPred).map { ($0 - $1).pow(2) }.reduce(0, +)
    print("\t loss \(loss.data)")
    loss.data
    // zero out the old grads
    for p in mlp.parameters {
        p.grad = 0.0
    }
    
    // backward pass
    backward(loss)
    
    // update
    let stepSize = 0.05
    for p in mlp.parameters {
        p.data += p.grad * -stepSize
    }
}


