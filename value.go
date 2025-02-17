package gograd

type Value struct {
	Data       float64
	Grad       float64
	Parents    []*Value
	backwardFn func(value *Value)
}

func NewValue(data float64) *Value {
	return &Value{
		Data: data,
	}
}
