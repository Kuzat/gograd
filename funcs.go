package gograd

import "math"

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

func (v *Value) Tanh() *Value {
	t := math.Tanh(v.Data)
	out := NewValue(t)
	out.Parents = []*Value{v}

	out.backwardFn = func(child *Value) {
		// child = tanh(v)
		// d(child)/d(v) = 1-tanh(v)^2 //ref: https://en.wikipedia.org/wiki/Hyperbolic_functions#Derivatives
		v.Grad += (1 - math.Pow(t, 2)) * child.Grad
	}

	return out
}

func (v *Value) Sigmoid() *Value {
	// out = 1 /(1 - e^(-x))
	return NewValue(1.0).Div(NewValue(1.0).Add(v.Neg().Exp()))
}

func Softmax(logits []*Value) []*Value {
	// softmax(zi) = e^(zi)/sum_j(e^(zj))
	exps := make([]*Value, len(logits))
	sumExps := NewValue(0)
	for i, logit := range logits {
		e := logit.Exp()
		exps[i] = e
		sumExps = sumExps.Add(e)
	}

	// normalize
	out := make([]*Value, len(logits))
	for i, e := range exps {
		out[i] = e.Div(sumExps)
	}

	return out
}

func CrossEntropyLoss(probs []*Value, targets []int) *Value {
	targetProbs := make([]*Value, len(targets))
	for i, target := range targets {
		targetProbs[i] = probs[target]
	}

	var totalLoss *Value = NewValue(0.0)
	for _, targetProb := range targetProbs {
		loss := targetProb.Log().Neg()
		totalLoss = totalLoss.Add(loss)
	}

	return totalLoss
}
