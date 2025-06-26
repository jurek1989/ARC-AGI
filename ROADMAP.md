# 🧩 ARC AGI SOLVER - ROADMAP

## 🎯 **CEL NADRZĘDNY:** Generalizacja na zadaniach ARC
System musi rozwiązywać nowe zadania ARC na podstawie analizy wzorców, nie przez dopasowanie do znanych przypadków.

---

## 📊 **STAN AKTUALNY KODU**

### ✅ **CO MAMY (Gotowe komponenty):**

#### **Podstawowa infrastruktura:**
- `Grid` class - operacje na siatkach (flip, rotate, crop, recolor, pad)
- `GridObject` class - wykrywanie obiektów z cechami (bbox, area, holes, colors, shape_type)
- `extract_all_object_views()` - 4 widoki obiektów (conn4/8, color/multicolor)
- `compare_grids()` - porównanie input/output (histogramy, rozmiary, tiling)

#### **DSL - Operacje na gridach:**
- `CropGrid`, `ResizeGridTo` - wycinanie i skalowanie
- `Transform` - rotacje i flipy
- `Pattern2D` - wzorce kaflowe

#### **DSL - Operacje na obiektach:**
- `Translate`, `MoveToBorder`, `MoveToTouch` - przesuwanie obiektów
- `Recolor`, `ResizeTo` - zmiana kolorów i rozmiaru
- `Sequence` - kompozycja operacji

#### **Heurystyki (częściowo gotowe):**
- Pattern2D - powielanie z transformacjami
- Mask + Tile engine - stare heurystyki kaflowe
- Cut-out detection - wycinanie fragmentów
- Object extraction - wyciąganie obiektów

#### **LLM Integration:**
- `llm_api_client.py` - **W PEŁNI ZINTEGROWANY** ✅
- `LLMAPIClient`, `TaskAnalysis`, `ObjectDescription` - struktury danych
- **Endpoint:** `https://brighton-t-zen-postcard.trycloudflare.com/generate`
- **Format:** `{"prompt": "text"}` → `{"response": "text"}`
- **Model:** Qwen3-0.6B na Colab + FastAPI + Cloudflare Tunnel
- **Status:** DZIAŁA - przetestowany i gotowy do użycia
- Fallback mechanisms - działanie bez modelu
- Integracja z głównym solverem

#### **Narzędzia:**
- `task_viewer.py` - wizualizacja zadań
- `debug_task()` - debugowanie pojedynczych zadań
- Testy i walidacja

### ❌ **CO BRAKUJE (Do zrobienia):**

#### **Krytyczne braki:**
- Systematyczne porównywanie obiektów input↔output
- Zaawansowane heurystyki (przesuwanie, dodawanie obiektów)
- Beam search / planista symboliczny
- ~~Konfiguracja LLM API na Google Colab~~ ✅ **GOTOWE**
- Testy regresyjne

#### **Brakujące operacje DSL:**
- `Scale`, `Copy`, `Delete`, `Merge` obiektów
- Operacje na wielu obiektach jednocześnie
- Operacje warunkowe

---

## 🚀 **PLAN WYKONANIA**

### **FAZA 1: Uzupełnienie podstaw (1-2 dni)**
**Priorytet: WYSOKI**

#### 1.1 Dokończenie porównywania obiektów
- [ ] Testy `match_objects()` - funkcja istnieje, wymaga testów
- [ ] Analiza różnic między obiektami input/output
- [ ] Priorytetyzacja dopasowań na podstawie cech
- [ ] Wykrywanie nowych/usuniętych obiektów
- [ ] **INTEGRACJA LLM** - analiza obiektów przez LLM

#### 1.2 Rozbudowa DSL operacji na obiektach
- [ ] `Scale(obj, factor)` - skalowanie obiektu
- [ ] `Copy(obj, position)` - kopiowanie obiektu
- [ ] `Delete(obj)` - usuwanie obiektu
- [ ] `Merge(obj1, obj2)` - łączenie obiektów
- [ ] Testy wszystkich nowych operacji

### **FAZA 2: Zaawansowane heurystyki (2-3 dni)**
**Priorytet: ŚREDNI**

#### 2.1 Heurystyki przesuwania obiektów
- [ ] Wykrywanie przesunięć między input/output
- [ ] Proponowanie operacji `Translate`
- [ ] Wykrywanie przesunięć grup obiektów
- [ ] Testy na zbiorze treningowym

#### 2.2 Heurystyki dodawania/usuwania obiektów
- [ ] Wykrywanie nowych obiektów w output
- [ ] Proponowanie operacji `Copy`, `Delete`
- [ ] Wykrywanie wzorców dodawania
- [ ] Testy na zbiorze treningowym

### **FAZA 3: Search i integracja LLM (GOTOWE)**
**Priorytet: WYSOKI**

#### 3.1 Beam search / planista symboliczny
- [ ] Implementacja beam search
- [ ] Przeszukiwanie przestrzeni rozwiązań
- [ ] Priorytetyzacja na podstawie heurystyk
- [ ] Integracja z istniejącymi heurystykami

#### 3.2 ~~Konfiguracja LLM API~~ ✅ **GOTOWE**
- [x] Stworzenie serwera API na Google Colab
- [x] Implementacja endpointów: `/generate`
- [x] Testy z prawdziwym modelem Qwen3-0.6B
- [x] Integracja z głównym solverem

### **FAZA 4: Fallback i optymalizacja (1-2 dni)**
**Priorytet: NISKI**

#### 4.1 Fallback strategie
- [ ] LLM generuje nowe operacje DSL
- [ ] Brute force w ograniczonym zakresie
- [ ] Losowe wariacje z walidacją
- [ ] Testy regresyjne dla nowych operacji

#### 4.2 Testy regresyjne i optymalizacja
- [ ] Pełne testy na zbiorze treningowym
- [ ] Optymalizacja wydajności
- [ ] Benchmarking różnych strategii
- [ ] Dokumentacja najlepszych praktyk

---

## 📈 **METRYKI SUKCESU**

### **Krótkoterminowe (po każdej fazie):**
- Liczba rozwiązywanych zadań na zbiorze treningowym
- Czas wykonania pojedynczego zadania
- Stabilność (brak regresji)

### **Długoterminowe:**
- Generalizacja na nowe zadania
- Skuteczność w konkursie Kaggle
- Możliwość rozszerzania o nowe operacje

---

## 🛠 **NAD CZYM OBECNIE PRACUJEMY**

**Status:** LLM ZINTEGROWANY - Przejście do Fazy 1.1
- [x] Usunięto lokalny model LLM (za ciężki dla laptopa)
- [x] Stworzono API client (llm_api_client.py)
- [x] Zintegrowano z głównym solverem
- [x] **LLM API w pełni funkcjonalne** ✅
- [x] **Endpoint:** `https://brighton-t-zen-postcard.trycloudflare.com/generate`
- [x] **Model:** Qwen3-0.6B na Colab + FastAPI + Cloudflare Tunnel
- [x] **Testy przechodzą** - LLM odpowiada na zapytania
- [x] Fallback mechanism działa (solver bez LLM)

**Następny krok:** Faza 1.1 - Dokończenie porównywania obiektów z integracją LLM

---

## 📝 **ZASADY ROZWOJU**

1. **Testy regresyjne** - po każdej zmianie sprawdzaj czy nie zepsułeś istniejącej funkcjonalności
2. **Iteracyjne podejście** - małe kroki, częste testy
3. **Dokumentacja** - każda nowa funkcja musi być udokumentowana
4. **Generalizacja** - priorytet nad dopasowaniem do konkretnych zadań
5. **Wydajność** - pamiętaj o ograniczeniach Kaggle (CPU-only, 12h limit)

---

## 🔄 **AKTUALIZACJE**

- **2024-12-19:** Utworzenie roadmapy
- **2024-12-19:** Model Qwen3-0.6B pobrany i gotowy do testów
- **2024-12-19:** Moduł LLM zintegrowany z kodem
- **2024-12-19:** Usunięto lokalny model LLM (za ciężki), stworzono API skeleton
- **2024-12-19:** Przejście do Fazy 1.1 - dokończenie porównywania obiektów
- **2024-12-19:** **LLM W PEŁNI ZINTEGROWANY** ✅ - FastAPI + Cloudflare Tunnel

---

*Ten dokument powinien być aktualizowany po każdej znaczącej zmianie w projekcie.* 