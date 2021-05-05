use ::foo::foofun
use ::fiz::fizfun

task Mapper(f: i32 -> i32) (In(i32)) -> (Out(i32))

  on In(x) => emit Out(x)

end

fun main()
  val x = foofun();
  val y = Mapper(|x| x + 1);
end
