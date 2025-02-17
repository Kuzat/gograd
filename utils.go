package gograd

func topSort(v *Value, visited map[*Value]bool, order *[]*Value) {
	if visited[v] {
		return
	}
	visited[v] = true
	for _, p := range v.Parents {
		topSort(p, visited, order)
	}
	*order = append(*order, v)
}
