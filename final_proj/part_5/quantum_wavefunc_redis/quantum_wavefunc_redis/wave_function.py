import redis
import numpy as np
import signal
import sys
import time
from typing import Tuple, Dict, Optional

# Redis配置类
class QuantumRedisConfig:
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_DB = 0
    REDIS_TIMEOUT = 15
    ROOT_HASH_KEY = "quantum:wave_func:psi_map"
    MAX_MEMORY_MB = 8000
    TEMP_DATA_TTL = 3600
    DEFAULT_GRID = (64, 64, 64)

CONFIG = QuantumRedisConfig()
redis_client: Optional[redis.Redis] = None
EXIT_FLAG = False

# 优雅退出处理函数
def graceful_exit(signum, frame):
    global EXIT_FLAG
    if EXIT_FLAG: return
    EXIT_FLAG = True
    print("\nStart graceful exit...")
    try:
        if redis_client and redis_client.connection_pool:
            redis_client.connection_pool.disconnect()
        clean_temp_data()
        print("Exit completed safely.")
    except Exception as e:
        print(f"Exit error: {str(e)}")
    finally:
        sys.exit(0)

# 注册退出信号监听
signal.signal(signal.SIGINT, graceful_exit)
signal.signal(signal.SIGTERM, graceful_exit)

# 初始化Redis连接与配置
def init_redis():
    global redis_client
    r = redis.Redis(
        host=CONFIG.REDIS_HOST,
        port=CONFIG.REDIS_PORT,
        db=CONFIG.REDIS_DB,
        decode_responses=False,
        socket_timeout=CONFIG.REDIS_TIMEOUT
    )
    r.ping()
    r.config_set("maxmemory", f"{CONFIG.MAX_MEMORY_MB}mb")
    r.config_set("maxmemory-policy", "allkeys-lru")
    r.config_set("hash-max-ziplist-entries", "10000")
    r.config_set("hash-max-ziplist-value", "64")
    redis_client = r
    return r

# 检查Redis内存使用率
def check_memory_usage():
    if not redis_client: return 0.0
    mem = redis_client.info("memory")
    used = mem["used_memory"] / 1024 / 1024
    usage = (used / CONFIG.MAX_MEMORY_MB) * 100
    if usage >= 85:
        print(f"Memory warning: {usage:.1f}%, clean temp data")
        clean_temp_data()
    return usage

# 清理临时数据
def clean_temp_data():
    if not redis_client: return
    temp_fields = [k for k, _ in redis_client.hscan_iter(CONFIG.ROOT_HASH_KEY, match=b"temp_*", count=1000)]
    if temp_fields: redis_client.hdel(CONFIG.ROOT_HASH_KEY, *temp_fields)
    temp_keys = redis_client.keys(pattern=b"quantum:temp:*")
    if temp_keys: redis_client.delete(*temp_keys)

# 生成三维波函数数据
def generate_wave_func(grid_shape: Tuple[int, int, int] = CONFIG.DEFAULT_GRID, t_step: float = 0.01):
    x = np.linspace(-5, 5, grid_shape[0])
    y = np.linspace(-5, 5, grid_shape[1])
    z = np.linspace(-5, 5, grid_shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    r2 = X**2 + Y**2 + Z**2
    psi = np.exp(-0.2 * r2 - 1j * 2 * np.pi * t_step)
    return psi.astype(np.complex128)

# 波函数Redis Hash操作处理器
class WaveFuncHashHandler:
    def __init__(self, client: redis.Redis):
        self.r = client
        self.root = CONFIG.ROOT_HASH_KEY

    # 复数转二进制字符串
    def c2b(self, c: complex) -> bytes:
        return f"{c.real:.10f},{c.imag:.10f}".encode()

    # 二进制字符串转复数
    def b2c(self, b: bytes) -> complex:
        r, i = b.decode().split(",")
        return complex(float(r), float(i))

    # 生成Hash字段名
    def gen_field(self, x: int, y: int, z: int, t: float, temp: bool = False) -> bytes:
        pre = b"temp_" if temp else b"formal_"
        return pre + f"x{x}_y{y}_z{z}_t{t:.4f}".encode()

    # 批量存储波函数数据到Hash
    def batch_save(self, psi: np.ndarray, t: float, temp: bool = False) -> Dict:
        st = time.time()
        nx, ny, nz = psi.shape
        data = {}
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    f = self.gen_field(x, y, z, t, temp)
                    data[f] = self.c2b(psi[x,y,z])
        self.r.hmset(self.root, data)
        if temp: self.r.expire(self.root, CONFIG.TEMP_DATA_TTL)
        return {"count": len(data), "cost": round(time.time()-st,2), "temp": temp}

    # 读取指定坐标点的波函数值
    def get_point(self, x: int, y: int, z: int, t: float, temp: bool = False) -> complex:
        f = self.gen_field(x, y, z, t, temp)
        val = self.r.hget(self.root, f)
        if not val: raise KeyError(f"Data not found ({x},{y},{z}) t={t}")
        return self.b2c(val)

    # 获取Hash数据统计信息
    def get_stats(self) -> Dict:
        field_cnt = self.r.hlen(self.root)
        mem = self.r.memory_usage(self.root) / 1024 / 1024
        sys_usage = check_memory_usage()
        return {
            "hash_key": self.root,
            "total_points": field_cnt,
            "mem_usage_mb": round(mem,2),
            "sys_usage_pct": round(sys_usage,1)
        }

# 计算量子力学可观测量
def calc_observables(psi: complex) -> Dict:
    mod = np.abs(psi)
    prob = mod ** 2
    phase = np.angle(psi, deg=True)
    return {
        "psi": psi,
        "modulus": round(mod,6),
        "prob_density": round(prob,6),
        "phase_deg": round(phase,2)
    }

# 主程序入口
def main():
    try:
        r = init_redis()
        handler = WaveFuncHashHandler(r)
        t0 = 0.01
        psi = generate_wave_func(time_step=t0)
        print(f"Wave func generated: shape={psi.shape}, dtype={psi.dtype}")

        res = handler.batch_save(psi, t0, temp=False)
        print(f"Batch saved: {res['count']} points, cost {res['cost']}s")

        x,y,z = 32,32,32
        psi_p = handler.get_point(x,y,z,t0)
        print(f"Point ({x},{y},{z}) t={t0}: {psi_p:.6f}")

        obs = calc_observables(psi_p)
        print("Observables:", obs)
        print("Stats:", handler.get_stats())

        print("\nRunning, press Ctrl+C to exit...")
        while not EXIT_FLAG:
            time.sleep(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        graceful_exit(signal.SIGINT, None)