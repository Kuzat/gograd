package gograd

import "math"

func (v *Value) Sub(other *Value) *Value {
	out := NewValue(v.Data - other.Data)
	out.Parents = []*Value{v, other}

	out.backwardFn = func(child *Value) {
		// d(child)/d(v) = 1
		v.Grad += child.Grad
		// d(child)/d(other) = -1
		other.Grad -= child.Grad
	}

	return out
}

func (v *Value) Add(other *Value) *Value {
	out := NewValue(v.Data + other.Data)
	out.Parents = []*Value{v, other}

	// backwardFn for the "add" (+)
	out.backwardFn = func(child *Value) {
		// child = out = v + other
		// d(child)/d(v) = 1, so v.Grad += child.Grad
		v.Grad += child.Grad
		// d(child)/d(other) = 1
		other.Grad += child.Grad
	}

	return out
}

func (v *Value) Mul(other *Value) *Value {
	out := NewValue(v.Data * other.Data)
	out.Parents = []*Value{v, other}

	out.backwardFn = func(child *Value) {
		// d(child)/d(v) = other.Data
		v.Grad += other.Data * child.Grad
		// d(child)/d(other) = v.Data
		other.Grad += v.Data * child.Grad
	}

	return out
}

func (v *Value) Div(other *Value) *Value {
	out := NewValue(v.Data / other.Data)
	out.Parents = []*Value{v, other}

	out.backwardFn = func(child *Value) {
		// child = v/other
		// d(child)/d(v) = 1 / other.Data
		v.Grad += (1.0 / other.Data) * child.Grad
		// d(child)/d(other) = -V.Data / (other.Data)^2
		other.Grad += (-v.Data / (other.Data * other.Data)) * child.Grad
	}

	return out
}

func (v *Value) Exp() *Value {
	// out = e^v
	e := math.Exp(v.Data)
	out := NewValue(e)
	out.Parents = []*Value{v}

	out.backwardFn = func(child *Value) {
		// child = out = e^(v)
		// d(child)/d(v) = e^(v)
		v.Grad += e * child.Grad
	}

	return out
}

func (v *Value) Neg() *Value {
	return NewValue(0.0).Sub(v)
}

func (v *Value) Log() *Value {
	// out = log(v)
	out := NewValue(math.Log(v.Data))
	out.Parents = []*Value{v}

	out.backwardFn = func(child *Value) {
		// child = out = log(v)
		// d(child)/d(v) = 1/v
		v.Grad += 1 / v.Data * child.Grad
	}

	return out
}
