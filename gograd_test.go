package gograd

import (
	"math"
	"testing"
)

func TestAdd(t *testing.T) {
	a := NewValue(2.0)
	b := NewValue(-3.0)
	c := a.Add(b) // c = 2 + (-3) = -1

	c.Backward()
	if c.Data != -1.0 {
		t.Errorf("Expected c.Data = -1, got %v", c.Data)
	}
	if a.Grad != 1.0 {
		t.Errorf("Expected a.Grad = 1.0, got %v", a.Grad)
	}
	if b.Grad != 1.0 {
		t.Errorf("Expected b.Grad = 1.0, got %v", b.Grad)
	}
}

func TestMul(t *testing.T) {
	x := NewValue(3.0)
	y := NewValue(4.0)
	z := x.Mul(y) // z = x*y = 12

	z.Backward()
	if math.Abs(x.Grad-4.0) > 1e-9 {
		t.Errorf("Expected x.Grad = 4.0, got %v", x.Grad)
	}
	if math.Abs(y.Grad-3.0) > 1e-9 {
		t.Errorf("Expected y.Grad = 3.0, got %v", y.Grad)
	}
}

func TestComplexExpression(t *testing.T) {
	// f(a,b) = a + 2*b
	a := NewValue(1.0)
	b := NewValue(2.0)

	// f = 1 + 2*2 = 5
	f := a.Add(b.Mul(NewValue(2.0)))
	f.Backward()

	// df/da = 1
	if math.Abs(a.Grad-1.0) > 1.e-9 {
		t.Errorf("Expected a.Grad = 1.0, got %v", a.Grad)
	}
	// df/db = 2
	if math.Abs(b.Grad-2.0) > 1.e-9 {
		t.Errorf("Expected b.Grad = 2.0, got %v", b.Grad)
	}
}

func TestValue_Tanh(t *testing.T) {
	v := NewValue(0.0)
	out := v.Tanh()
	out.Backward()

	if math.Abs(out.Data-math.Tanh(0.0)) > 1e-9 {
		t.Errorf("Expected out.Data = tanh(0.0), got %v", out.Data)
	}

	if math.Abs(v.Grad-1.0) > 1e-9 {
		t.Errorf("Expected v.Grad = 1.0, got %v", v.Grad)
	}
}

func TestValue_Exp(t *testing.T) {
	v := NewValue(0.0)
	out := v.Exp()
	out.Backward()

	if math.Abs(out.Data-1.0) > 1e-9 {
		t.Errorf("Expected out.Data = 1.0, got %v", out.Data)
	}

	if math.Abs(v.Grad-1.0) > 1e-9 {
		t.Errorf("Expected v.Grad = 1.0, got %v", v.Grad)
	}
}

func TestValue_Neg(t *testing.T) {
	v := NewValue(1.0)
	out := v.Neg()
	out.Backward()

	if out.Data != -1.0 {
		t.Errorf("Expected out.Data = -1.0, got %v", out.Data)
	}

	if v.Grad != -1.0 {
		t.Errorf("Expected v.Grad = -1.0, got %v", v.Grad)
	}
}

func TestValue_Sigmoid(t *testing.T) {
	v := NewValue(0.0)
	out := v.Sigmoid()
	out.Backward()

	if math.Abs(out.Data-0.5) > 1e-9 {
		t.Errorf("Expected out.Data = 0.5, got %v", out.Data)
	}

	if math.Abs(v.Grad-0.25) > 1e-9 {
		t.Errorf("Expected v.Grad = 0.25, got %v", v.Grad)
	}
}

func TestValue_Softmax(t *testing.T) {
	logits := []*Value{
		NewValue(1.0),
		NewValue(2.0),
		NewValue(3.0),
	}
	probs := Softmax(logits)
	for _, p := range probs {
		p.Backward()
	}

	expected := []float64{
		0.09003057317038046,
		0.24472847105479767,
		0.6652409557748219,
	}
	var sum float64
	for i, p := range probs {
		if math.Abs(p.Data-expected[i]) > 1e-9 {
			t.Errorf("Expected probs[%d].Data = %v, got %v", i, expected[i], p.Data)
		}
		sum += p.Data
	}
	if math.Abs(sum-1.0) > 1e-9 {
		t.Errorf("Expected sum of probabilities = 1.0, got %v", sum)
	}
}

func TestValue_Log(t *testing.T) {
	v := NewValue(1.0)
	out := v.Log()
	out.Backward()

	if math.Abs(out.Data-0.0) > 1e-9 {
		t.Errorf("Expected out.Data = 0.0, got %v", out.Data)
	}

	if math.Abs(v.Grad-1.0) > 1e-9 {
		t.Errorf("Expected v.Grad = 1.0, got %v", v.Grad)
	}
}

func TestCrossEntropyLoss(t *testing.T) {
	probs := []*Value{
		NewValue(0.1),
		NewValue(0.8),
		NewValue(0.1),
	}
	targets := []int{1}
	loss := CrossEntropyLoss(probs, targets)
	loss.Backward()

	if math.Abs(loss.Data+math.Log(0.8)) > 1e-9 {
		t.Errorf("Expected loss.Data = %v, got %v", math.Log(0.8), loss.Data)
	}

	expected := []float64{0.0, -1.25, 0.0}
	for i, p := range probs {
		if math.Abs(p.Grad-expected[i]) > 1e-9 {
			t.Errorf("Expected probs[%d].Grad = %v, got %v", i, expected[i], p.Grad)
		}
	}
}
