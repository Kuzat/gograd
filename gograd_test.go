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
