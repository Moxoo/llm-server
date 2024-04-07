// <auto-generated>
//  automatically generated by the FlatBuffers compiler, do not modify
// </auto-generated>

using global::System;
using global::FlatBuffers;

public struct Attacker : IFlatbufferObject
{
  private Table __p;
  public ByteBuffer ByteBuffer { get { return __p.bb; } }
  public static Attacker GetRootAsAttacker(ByteBuffer _bb) { return GetRootAsAttacker(_bb, new Attacker()); }
  public static Attacker GetRootAsAttacker(ByteBuffer _bb, Attacker obj) { return (obj.__assign(_bb.GetInt(_bb.Position) + _bb.Position, _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __p.bb_pos = _i; __p.bb = _bb; }
  public Attacker __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public int SwordAttackDamage { get { int o = __p.__offset(4); return o != 0 ? __p.bb.GetInt(o + __p.bb_pos) : (int)0; } }

  public static Offset<Attacker> CreateAttacker(FlatBufferBuilder builder,
      int sword_attack_damage = 0) {
    builder.StartObject(1);
    Attacker.AddSwordAttackDamage(builder, sword_attack_damage);
    return Attacker.EndAttacker(builder);
  }

  public static void StartAttacker(FlatBufferBuilder builder) { builder.StartObject(1); }
  public static void AddSwordAttackDamage(FlatBufferBuilder builder, int swordAttackDamage) { builder.AddInt(0, swordAttackDamage, 0); }
  public static Offset<Attacker> EndAttacker(FlatBufferBuilder builder) {
    int o = builder.EndObject();
    return new Offset<Attacker>(o);
  }
};
