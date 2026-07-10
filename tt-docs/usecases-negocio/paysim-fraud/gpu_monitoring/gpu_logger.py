"""
Logger de metricas de GPU durante o treino (temperatura, VRAM, uso%, watts).
Baseado no snippet de pynvml do script-desafio.md, empacotado como classe
reutilizavel + gravacao em CSV para o gpu_report.md depois.

Uso dentro do loop de treino:

    from gpu_logger import GPULogger

    logger = GPULogger(csv_path="gpu_log.csv")
    for epoca in range(epocas):
        ...
        logger.registrar(epoca=epoca, loss=perda.item())

    logger.fechar()
"""

import csv
import time
from pathlib import Path


class GPULogger:
    def __init__(self, csv_path: str = "gpu_log.csv", indice_gpu: int = 0):
        self.csv_path = Path(csv_path)
        self.indice_gpu = indice_gpu
        self._disponivel = self._iniciar_pynvml()
        self._arquivo = None
        self._writer = None
        self._criar_csv()

    def _iniciar_pynvml(self) -> bool:
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.indice_gpu)
            return True
        except Exception as e:
            print(f"[gpu_logger] pynvml indisponivel ({e}) - "
                  f"metricas de GPU nao serao coletadas")
            return False

    def _criar_csv(self):
        novo = not self.csv_path.exists()
        self._arquivo = open(self.csv_path, "a", newline="")
        self._writer = csv.writer(self._arquivo)
        if novo:
            self._writer.writerow([
                "timestamp", "epoca", "loss", "vram_usada_gb", "vram_total_gb",
                "temperatura_c", "uso_gpu_pct", "consumo_watts",
            ])

    def stats(self) -> dict:
        if not self._disponivel:
            return {
                "vram_usada_gb": None, "vram_total_gb": None,
                "temperatura_c": None, "uso_gpu_pct": None, "consumo_watts": None,
            }
        mem = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        temp = self.pynvml.nvmlDeviceGetTemperature(
            self.handle, self.pynvml.NVML_TEMPERATURE_GPU
        )
        util = self.pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        power = self.pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000
        return {
            "vram_usada_gb": mem.used / 1e9,
            "vram_total_gb": mem.total / 1e9,
            "temperatura_c": temp,
            "uso_gpu_pct": util.gpu,
            "consumo_watts": power,
        }

    def registrar(self, epoca: int, loss: float | None = None):
        s = self.stats()
        self._writer.writerow([
            time.time(), epoca, loss,
            s["vram_usada_gb"], s["vram_total_gb"],
            s["temperatura_c"], s["uso_gpu_pct"], s["consumo_watts"],
        ])
        self._arquivo.flush()

        if s["temperatura_c"] is not None and s["temperatura_c"] > 83:
            print(f"[gpu_logger] AVISO: temperatura {s['temperatura_c']}C "
                  f"acima do limite recomendado (83C) - risco de throttling")
        return s

    def fechar(self):
        if self._arquivo:
            self._arquivo.close()


if __name__ == "__main__":
    # teste rapido: 5 leituras espacadas de 1s
    logger = GPULogger(csv_path="gpu_log_teste.csv")
    for i in range(5):
        stats = logger.registrar(epoca=i)
        print(stats)
        time.sleep(1)
    logger.fechar()
