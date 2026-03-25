# Music Flamingo Local GUI

Ez a projekt egy helyi webes feluletet ad a `nvidia/music-flamingo-think-2601-hf` modellhez.
Az app audio feltoltest, szoveges promptot, tobbfordulos beszelgetest es valaszparameter-allitasokat ad.

## Attekintes

A projekt celja, hogy egy egyszeru, helyi kezelofeluleten keresztul tudd hasznalni a Music Flamingo modellt Windows rendszeren.
A felulet a modell betolteset, az audiofajlok feltolteset es a valaszok finomhangolasat kezeli.

## Fontos technikai megjegyzes a kvantalasrol

A celmodell jelenleg nem egy sima `llama.cpp`-kompatibilis szovegmodell.
A Hugging Face konfiguracioja szerint az architekturaja `MusicFlamingoForConditionalGeneration`, benne kulon `audioflamingo3_encoder` resszel es Qwen2 szovegmaggal.

Ez a gyakorlatban azt jelenti, hogy:

- a modellhez jelenleg nincs ismert, mukodo `GGUF` konverzios ut `llama.cpp`-hez
- a realis memoriaoptimalizalt helyi futtatasi ut jelenleg a Transformers-alapu 4 bites `NF4` betoltes

A GUI-ban ezert a `4-bit NF4` opcio van implementalva. Ez nem `Q4_0` vagy `Q4_K_M` GGUF, hanem `BitsAndBytesConfig(load_in_4bit=True)`.

## Rendszerigeny

- Windows
- NVIDIA GPU CUDA tamogatassal
- ajanlott: RTX 4090 Laptop GPU vagy erosebb
- Python 3.11
- eleg szabad lemezterulet a 16.5 GB-os sulyokhoz es a cache-hez

## Telepites

1. Hozz letre virtualis kornyezetet:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Ha a rendszereden van kulon Python 3.11 telepites es a Windows launcher latja, akkor ez is jo lehet:

```powershell
py -3.11 -m venv .venv
```

Ha ez a parancs azt mondja, hogy `No suitable Python runtime found`, akkor nem a Python hianyzik, hanem a `py` launcher nem lat 3.11-es regisztralt runtime-ot. Ilyenkor hasznald a `python -m venv .venv` formát.

2. Telepits CUDA-s PyTorch buildet a sajat CUDA verziodhoz illo modon a PyTorch hivatalos utmutatoja szerint.

Pelda CUDA 12.4-re:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

13.0:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

3. Telepitsd a tobbi fuggoseget:

```powershell
pip install -r requirements.txt
```

Az audio input miatt a `librosa` is kell; ezt a requirements mar tartalmazza.

4. Ha a Hugging Face oldal licencelfogadast ker, azt a modell oldalan fogadd el, es szukseg eseten jelentkezz be:

```powershell
huggingface-cli login
```

## Inditas

Kezi inditas:

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

Vagy a PowerShell inditoszkripttel:

```powershell
.\start.ps1
```

Elso telepitessel egyutt:

```powershell
.\start.ps1 -Install
```

Ha a gépeden a `py` launcher nem talál megfelelő runtime-ot, a `start.ps1` mar a helyi `python` parancsra támaszkodik, ezért az Anaconda/base vagy bármely elérhető interpreterrel is tud virtualis kornyezetet hozni.

Az alkalmazas a bongeszoben a `http://127.0.0.1:7860` cimen nyilik meg.

## Hasznalat

1. Valaszd ki a `4-bit NF4` vagy `BF16 / FP16` modot.
2. Kattints a `Modell betoltese` gombra.
3. Adj meg egy promptot.
4. Opcionlisan tolts fel MP3, WAV vagy FLAC fajlt.
5. Kuldd el a kerdest.

## Licenc

Ez a projekt MIT licenc alatt all. Reszletek a [LICENSE](LICENSE) fajlban.

## Praktikus beallitasok 4090 Laptop GPU-hoz

- kezdesnek maradj `4-bit NF4` modban
- a projekt alapbol CPU offloaddal es 12 GiB GPU limittel probalja betolteni a modellt, mert a teljes Music Flamingo gyakran nem fer be 16 GB laptop VRAM-ba
- hasznalj rovidebb audio reszleteket teszthez
- allits `max_new_tokens` erteket inkabb 128-192 koze; audioval a modell kontextuskerete gyorsan megtelik
- ha VRAM hibaba futsz, csokkentsd a tokenlimitet es zard be a tobbi GPU-t hasznalo appot
- ha audio betoltesi hibat latsz, ellenorizd hogy a `librosa` telepitve van-e a virtualis kornyezetben