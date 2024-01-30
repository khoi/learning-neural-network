import Cocoa
import Foundation

final class Value {
    enum Operator: CustomStringConvertible {
        case add
        case multiply
        case tanh
        
        var description: String {
            switch self {
            case .add:
                "+"
            case .multiply:
                "*"
            case .tanh:
                "tanh"
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

extension Value: ExpressibleByFloatLiteral {
    convenience init(floatLiteral value: Double) {
        self.init(value)
    }
}

extension Value: CustomStringConvertible {
    var description: String {
        String(format: "%@ | data %.4f | grad: %.4f", label ?? "", data, grad)
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
    func tanh() -> Value {
        let t = Darwin.tanh(data)
        let out = Value(
            t,
            children: [self],
            op: .tanh
        )
        
        out.backward = { [unowned self] in
            self.grad += (1 - pow(t, 2)) * out.grad
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

func traverse(_ v: Value) {
    print(v)
    v.children.forEach {
        traverse($0)
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

var out = n.tanh()
out.label = "o"

backward(out)
traverse(out)