package gograd

func (v *Value) Backward() {
	// 1. Set the gradient of the output node (v) to 1
	v.Grad = 1

	// 2. Build a topological order
	var order []*Value
	visited := make(map[*Value]bool)
	topSort(v, visited, &order)

	// 3. Travers in reverse order, calling backwardFN
	for i := len(order) - 1; i >= 0; i-- {
		node := order[i]
		if node.backwardFn != nil {
			node.backwardFn(node)
		}
	}
}

func (v *Value) ZeroGrad() {
	var order []*Value
	visited := make(map[*Value]bool)
	topSort(v, visited, &order)

	for _, node := range order {
		node.Grad = 0.0
	}
}
