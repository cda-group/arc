# XFAIL: *
# RUN: arc-lang %s

val a = 0;
val b = a;
def c() = a;
def d() = c();
type T = i32;

def e() {
  val x = " a + 1 = ${a + 1} ";
  val a = a;
  val b = a;
  def c() = a;
  def d() = c();
  d();
}

def f0(a) = a+1;
val f1 = fun(a) = a+1;

a.f0();
a.f1();
a.f1();
a.f1();
a.f1();

def r0() = r1();
def r1() = r0();
