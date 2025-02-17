package gograd

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
		// d(child)/d(v) = 1 / other.Data
		v.Grad += (1.0 / other.Data) * child.Grad
		// d(child)/d(other) = -V.Data / (other.Data)^2
		other.Grad += (-v.Data / (other.Data * other.Data)) * child.Grad
	}

	return out
}

func (v *Value) ReLU() *Value {
	var out *Value
	if v.Data > 0 {
		out = NewValue(v.Data)
	} else {
		out = NewValue(0)
	}
	out.Parents = []*Value{v}

	out.backwardFn = func(child *Value) {
		// child = v if v > 0 else 0
		// d(child)/d(v) = 1 if v > 0 else 0
		if v.Data > 0 {
			v.Grad += child.Grad
		}
	}

	return out
}
