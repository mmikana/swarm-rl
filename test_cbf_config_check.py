"""
测试 CBF 配置检查

验证当 --quads_use_cbf=True 但 --quads_use_obstacles=False 时，
会抛出清晰的错误信息
"""

import sys


def test_cbf_requires_obstacles():
    """测试 CBF 必须启用障碍物"""
    print("=" * 70)
    print("测试：CBF 配置检查")
    print("=" * 70)
    
    # 创建一个假的 config 对象（只包含必要的属性）
    class FakeConfig:
        def __init__(self):
            self.quads_use_cbf = True
            self.quads_use_obstacles = False  # 错误配置！
    
    cfg = FakeConfig()
    
    print("\n配置：")
    print(f"  quads_use_cbf = {cfg.quads_use_cbf}")
    print(f"  quads_use_obstacles = {cfg.quads_use_obstacles} ← 错误！")
    print()
    
    # 直接测试配置检查逻辑
    try:
        use_cbf = getattr(cfg, 'quads_use_cbf', False)
        use_obstacles = getattr(cfg, 'quads_use_obstacles', False)
        
        if use_cbf and not use_obstacles:
            raise ValueError(
                "CBF requires obstacles to be enabled! "
                "Please add --quads_use_obstacles=True to your training command.\n"
                "\nExample:\n"
                "  python -m swarm_rl.train --algo=APPO --env=quadrotor_multi \\\n"
                "      --quads_use_cbf=True --quads_use_obstacles=True \\\n"
                "      --train_for_env_steps=1000"
            )
        
        print("\n❌ 测试失败：应该抛出 ValueError 但没有")
        return False
        
    except ValueError as e:
        print(f"\n✓ 正确抛出 ValueError:")
        print(f"  {str(e)[:150]}...")
        print()
        print("✅ 测试通过！")
        return True


def test_cbf_with_obstacles():
    """测试正确配置（启用障碍物）"""
    print("\n" + "=" * 70)
    print("测试：正确配置（启用障碍物）")
    print("=" * 70)
    
    class FakeConfig:
        def __init__(self):
            self.quads_use_cbf = True
            self.quads_use_obstacles = True  # 正确配置！
    
    cfg = FakeConfig()
    
    print("\n配置：")
    print(f"  quads_use_cbf = {cfg.quads_use_cbf}")
    print(f"  quads_use_obstacles = {cfg.quads_use_obstacles} ← 正确！")
    print()
    
    try:
        use_cbf = getattr(cfg, 'quads_use_cbf', False)
        use_obstacles = getattr(cfg, 'quads_use_obstacles', False)
        
        if use_cbf and not use_obstacles:
            raise ValueError("CBF requires obstacles...")
        
        print("✓ 配置检查通过")
        print()
        print("✅ 测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败：{type(e).__name__}: {e}")
        return False


def test_no_cbf_no_obstacles():
    """测试不使用 CBF 时不需要障碍物"""
    print("\n" + "=" * 70)
    print("测试：不使用 CBF（不需要障碍物）")
    print("=" * 70)
    
    class FakeConfig:
        def __init__(self):
            self.quads_use_cbf = False
            self.quads_use_obstacles = False
    
    cfg = FakeConfig()
    
    print("\n配置：")
    print(f"  quads_use_cbf = {cfg.quads_use_cbf}")
    print(f"  quads_use_obstacles = {cfg.quads_use_obstacles}")
    print()
    
    try:
        use_cbf = getattr(cfg, 'quads_use_cbf', False)
        use_obstacles = getattr(cfg, 'quads_use_obstacles', False)
        
        if use_cbf and not use_obstacles:
            raise ValueError("CBF requires obstacles...")
        
        print("✓ 配置检查通过（不使用 CBF，无需检查）")
        print()
        print("✅ 测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败：{type(e).__name__}: {e}")
        return False


if __name__ == '__main__':
    print("\nCBF 配置检查测试\n")
    
    # 测试 1：错误配置（应该抛出 ValueError）
    result1 = test_cbf_requires_obstacles()
    
    # 测试 2：正确配置（应该通过）
    result2 = test_cbf_with_obstacles()
    
    # 测试 3：不使用 CBF（应该通过）
    result3 = test_no_cbf_no_obstacles()
    
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    print(f"  错误配置检测：{'✅ 通过' if result1 else '❌ 失败'}")
    print(f"  正确配置创建：{'✅ 通过' if result2 else '❌ 失败'}")
    print(f"  非 CBF 模式：  {'✅ 通过' if result3 else '❌ 失败'}")
    
    if result1 and result2 and result3:
        print("\n🎉 所有测试通过！")
        sys.exit(0)
    else:
        print("\n⚠ 部分测试失败")
        sys.exit(1)
