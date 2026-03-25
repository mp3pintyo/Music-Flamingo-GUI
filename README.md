# Music Flamingo Local GUI

Ez a projekt egy helyi webes felületet ad a `nvidia/music-flamingo-think-2601-hf` modellhez.
Az app audiofeltöltést, szöveges promptot, többfordulós beszélgetést és válaszparaméter-allításokat ad.

## Áttekintés

A projekt célja, hogy egy egyszerű, helyi kezelőfelületen keresztül tudd használni a Music Flamingo modellt Windows rendszeren.
A felület a modell betöltését, az audiofájlok feltöltését és a válaszok finomhangolását kezeli.

## Fontos technikai megjegyzés a kvantálásról

A célmodell jelenleg nem egy sima `llama.cpp`-kompatibilis szövegmodell.
A Hugging Face konfigurációja szerint az architektúrája `MusicFlamingoForConditionalGeneration`, benne külön `audioflamingo3_encoder` résszel és Qwen2 szövegmaggal.

Ez a gyakorlatban azt jelenti, hogy:

- a modellhez jelenleg nincs ismert, működő `GGUF` konverziós út `llama.cpp`-hez
- a reális memóriaoptimalizált helyi futtatási út jelenleg a Transformers-alapú 4 bites `NF4` betöltés

A GUI-ban ezért a `4-bit NF4` opció van implementálva. Ez nem `Q4_0` vagy `Q4_K_M` GGUF, hanem `BitsAndBytesConfig(load_in_4bit=True)`.

## Rendszerigény

- Windows
- NVIDIA GPU CUDA-támogatással
- ajánlott: RTX 4090 Laptop GPU vagy erősebb
- Python 3.11
- elég szabad lemezterület a 16.5 GB-os súlyokhoz és a cache-hez

## Telepítés

1. Hozz létre virtuális környezetet:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Ha a rendszereden van külön Python 3.11 telepítés, és a Windows launcher látja, akkor ez is jó lehet:

```powershell
py -3.11 -m venv .venv
```

Ha ez a parancs azt mondja, hogy `No suitable Python runtime found`, akkor nem a Python hiányzik, hanem a `py` launcher nem lát 3.11-es regisztrált runtime-ot. Ilyenkor használd a `python -m venv .venv` formát.

2. Telepíts CUDA-s PyTorch buildet a saját CUDA-verziódhoz illő módon a PyTorch hivatalos útmutatója szerint.

Pelda CUDA 12.4-re:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

13.0:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

3. Telepítsd a többi függőséget:

```powershell
pip install -r requirements.txt
```

Az audio bemenet miatt a `librosa` is kell; ezt a requirements már tartalmazza.

4. Ha a Hugging Face oldal licencelfogadást kér, azt a modell oldalán fogadd el, és szükség esetén jelentkezz be:

```powershell
huggingface-cli login
```

## Indítás

Kézi indítás:

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

Vagy a PowerShell indítószkripttel:

```powershell
.\start.ps1
```

Első telepítéssel együtt:

```powershell
.\start.ps1 -Install
```

Ha a gépeden a `py` launcher nem talál megfelelő runtime-ot, a `start.ps1` már a helyi `python` parancsra támaszkodik, ezért az Anaconda/base vagy bármely elérhető interpreterrel is tud virtuális környezetet hozni.

Az alkalmazás a böngészőben a `http://127.0.0.1:7860` címen nyílik meg.

## Használat

1. Válaszd ki a `4-bit NF4` vagy `BF16 / FP16` módot.
2. Kattints a `Modell betöltése` gombra.
3. Adj meg egy promptot.
4. Opcionálisan tölts fel MP3, WAV vagy FLAC fájlt.
5. Küldd el a kérdést.

## Licenc

Ez a projekt MIT licenc alatt áll. Részletek a [LICENSE](LICENSE) fájlban.

## Praktikus beállítások 4090 Laptop GPU-hoz

- kezdetben maradj `4-bit NF4` módban
- a projekt alapból CPU offloaddal és 12 GiB GPU limittel próbálja betölteni a modellt, mert a teljes Music Flamingo gyakran nem fér be 16 GB laptop VRAM-ba
- használj rövidebb audio részleteket teszthez
- állítsd a `max_new_tokens` értékét inkább 128-192 közé; audióval a modell kontextuskerete gyorsan megtelik
- ha VRAM-hibába futsz, csökkentsd a tokenlimitet és zárd be a többi GPU-t használó appot
- ha audio betöltési hibát látsz, ellenőrizd, hogy a `librosa` telepítve van-e a virtuális környezetben