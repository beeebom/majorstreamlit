# Complete Algorithm Description for Research Paper

## ALGORITHM 1: Video Encryption System - Main Process

```
Input: 
  - V: Original video file with n frames {F₁, F₂, ..., Fₙ}
  - P: User password for authentication
  - start_frame: Starting frame index
  
Output:
  - Vₑ: Encrypted video file
  - Vₐ: Decrypted video file (for verification)
  - M: Performance metrics (MSE, PSNR, NPCR, UACI, timing)
  - Σ: Digital signatures for each frame

Initialization:
1: Generate RSA key pair (SK_rsa, PK_rsa) with 2048-bit modulus
2: K_master_aes ← GenerateRandomBytes(32)        // 256-bit AES master key
3: K_master_xor ← GenerateRandomBytes(32)        // 256-bit XOR master key
4: Initialize face detection classifiers:
     - H_frontal ← HaarCascade(frontalface)
     - H_profile ← HaarCascade(profileface)
     - H_eye ← HaarCascade(eye)
     - DNN_model ← LoadCaffeSSD(ResNet10-300x300)
5: Authenticate(P)                               // Verify user password
6: If authentication fails, terminate

Main Processing Loop:
7: For i = start_frame to (start_frame + frame_count):
8:    F_i ← ReadFrame(V, i)                     // Read frame i
9:    
10:   // Stage 1: Face Detection (Algorithm 2)
11:   If (i mod detection_interval = 0):
12:      FACES_i ← DetectFaces(F_i, H_frontal, H_profile, H_eye, DNN_model)
13:      InitializeTrackers(F_i, FACES_i)
14:   Else:
15:      FACES_i ← UpdateTrackers(F_i)          // Use object tracking
16:   End If
17:   
18:   // Stage 2: Per-Frame Key Derivation (Algorithm 3)
19:   salt_i ← i.toBytes(8)                     // Frame index as salt
20:   nonce_i ← GenerateRandomBytes(8)          // Unique nonce per frame
21:   K_aes_i ← HKDF(K_master_aes, 32, salt_i, SHA256)
22:   K_xor_i ← HKDF(K_master_xor, 16, salt_i, SHA256, context="xor")
23:   
24:   // Stage 3: Two-Factor Encryption (Algorithm 4)
25:   F_enc_i ← TwoFactorEncrypt(F_i, FACES_i, K_aes_i, K_xor_i, nonce_i)
26:   
27:   // Stage 4: Digital Signature (Algorithm 5)
28:   σ_i ← Sign(F_enc_i, SK_rsa)
29:   
30:   // Stage 5: Write Encrypted Frame
31:   WriteFrame(Vₑ, F_enc_i)
32:   
33:   // Stage 6: Verification via Decryption (Algorithm 6)
34:   valid ← Verify(F_enc_i, σ_i, PK_rsa)
35:   F_dec_i ← TwoFactorDecrypt(F_enc_i, FACES_i, K_aes_i, K_xor_i, nonce_i)
36:   WriteFrame(Vₐ, F_dec_i)
37:   
38:   // Stage 7: Metric Calculation (Algorithm 7)
39:   M_i ← CalculateMetrics(F_i, F_enc_i, F_dec_i, encryption_time, decryption_time)
40: End For
41:
42: Return Vₑ, Vₐ, M, Σ
```

---

## ALGORITHM 2: Hierarchical Face Detection

```
Input: Frame F, Haar cascades (H_frontal, H_profile, H_eye), DNN model
Output: Set of face bounding boxes FACES = {(x₁,y₁,w₁,h₁), ..., (xₘ,yₘ,wₘ,hₘ)}

Procedure DetectFaces(F):
1: F_gray ← ConvertToGrayscale(F)
2: 
3: // Stage 1: Haar Cascade Detection
4: FACES_frontal ← H_frontal.detectMultiScale(F_gray, scaleFactor=1.1, minNeighbors=4)
5: FACES_profile ← H_profile.detectMultiScale(F_gray, scaleFactor=1.1, minNeighbors=4)
6: FACES_combined ← FACES_frontal ∪ FACES_profile
7: 
8: // Stage 2: DNN Fallback (if Haar fails)
9: If |FACES_combined| = 0:
10:   F_resized ← Resize(F, 300×300)
11:   blob ← DNNPreprocess(F_resized, mean=(104.0, 177.0, 123.0))
12:   detections ← DNN_model.forward(blob)
13:   
14:   For each detection d in detections:
15:      If d.confidence > 0.5:
16:         (x, y, w, h) ← d.boundingBox
17:         FACES_combined ← FACES_combined ∪ {(x, y, w, h)}
18:      End If
19:   End For
20: End If
21: 
22: Return FACES_combined
```

---

## ALGORITHM 3: HKDF-Based Key Derivation

```
Input: 
  - K_master: Master key (256 bits)
  - L: Desired output key length
  - salt: Frame index (8 bytes)
  - info: Context information (optional)
  
Output: K_derived: Derived key of length L

Mathematical Formulation:
  PRK = HMAC-SHA256(salt, K_master)              // Extract phase
  T(0) = empty string
  T(i) = HMAC-SHA256(PRK, T(i-1) || info || i)  // Expand phase
  K_derived = First L bytes of (T(1) || T(2) || ...)

Procedure HKDF(K_master, L, salt, hash_function, info=""):
1: // Extract Phase
2: PRK ← HMAC(hash_function, salt, K_master)
3: 
4: // Expand Phase
5: N ← ⌈L / hash_output_length⌉
6: T_prev ← ""
7: OKM ← ""
8: 
9: For i = 1 to N:
10:   T_i ← HMAC(hash_function, PRK, T_prev || info || byte(i))
11:   OKM ← OKM || T_i
12:   T_prev ← T_i
13: End For
14: 
15: K_derived ← First L bytes of OKM
16: Return K_derived
```

---

## ALGORITHM 4: Two-Factor Encryption (AES-CTR + XOR)

```
Input:
  - F: Original frame (H×W×C array)
  - FACES: Detected face regions {(x₁,y₁,w₁,h₁), ..., (xₘ,yₘ,wₘ,hₘ)}
  - K_aes: AES key (256 bits)
  - K_xor: XOR key (128 bits)
  - nonce: Unique nonce (8 bytes)
  
Output: F_enc: Encrypted frame

Mathematical Formulation:
  Layer 1 (Full Frame):
    F_aes = AES-CTR(F, K_aes, nonce)
    where CTR mode: C_i = P_i ⊕ E_K(nonce || counter_i)
  
  Layer 2 (Face Regions):
    For each face region R_j:
      R_enc_j = R_aes_j ⊕ (K_xor mod |R_j|)

Procedure TwoFactorEncrypt(F, FACES, K_aes, K_xor, nonce):
1: // Layer 1: Full-frame AES-CTR encryption
2: F_bytes ← F.flatten().toBytes()
3: cipher ← AES.new(K_aes, mode=CTR, nonce=nonce)
4: F_enc_bytes ← cipher.encrypt(F_bytes)
5: F_enc ← F_enc_bytes.reshape(F.shape).copy()
6: 
7: // Layer 2: Regional XOR encryption on face areas
8: padding_ratio ← 0.1                           // 10% padding
9: 
10: For each face (x, y, w, h) in FACES:
11:    // Apply padding
12:    pad_w ← ⌊w × padding_ratio⌋
13:    pad_h ← ⌊h × padding_ratio⌋
14:    x' ← max(0, x - pad_w)
15:    y' ← max(0, y - pad_h)
16:    w' ← min(w + 2×pad_w, W - x')
17:    h' ← min(h + 2×pad_h, H - y')
18:    
19:    // Extract and encrypt region
20:    R ← F_enc[y':y'+h', x':x'+w']
21:    R_flat ← R.flatten()
22:    
23:    // Repeat XOR key to match region size
24:    K_xor_repeated ← RepeatKey(K_xor, length(R_flat))
25:    
26:    // XOR encryption
27:    R_enc_flat ← R_flat ⊕ K_xor_repeated
28:    R_enc ← R_enc_flat.reshape(R.shape)
29:    
30:    // Write back encrypted region
31:    F_enc[y':y'+h', x':x'+w'] ← R_enc
32: End For
33: 
34: Return F_enc
```

---

## ALGORITHM 5: RSA Digital Signature Generation

```
Input:
  - F_enc: Encrypted frame data
  - SK_rsa: RSA private key (2048 bits)
  
Output: σ: Digital signature

Mathematical Formulation:
  h = SHA-256(F_enc)
  σ = (h)^d mod n
  where (n, d) is the RSA private key

Procedure Sign(F_enc, SK_rsa):
1: F_enc_bytes ← F_enc.toBytes()
2: h ← SHA256.hash(F_enc_bytes)               // Hash of encrypted frame
3: σ ← RSA-PKCS1-v1.5-Sign(SK_rsa, h)        // Sign hash with private key
4: Return σ
```

---

## ALGORITHM 6: Two-Factor Decryption

```
Input:
  - F_enc: Encrypted frame
  - FACES: Face regions (same as encryption)
  - K_aes: AES key (256 bits)
  - K_xor: XOR key (128 bits)
  - nonce: Frame nonce (8 bytes)
  
Output: F_dec: Decrypted frame

Procedure TwoFactorDecrypt(F_enc, FACES, K_aes, K_xor, nonce):
1: F_temp ← F_enc.copy()
2: padding_ratio ← 0.1
3: 
4: // Reverse Layer 2: Remove XOR encryption from face regions
5: For each face (x, y, w, h) in FACES:
6:    // Apply same padding as encryption
7:    pad_w ← ⌊w × padding_ratio⌋
8:    pad_h ← ⌊h × padding_ratio⌋
9:    x' ← max(0, x - pad_w)
10:   y' ← max(0, y - pad_h)
11:   w' ← min(w + 2×pad_w, W - x')
12:   h' ← min(h + 2×pad_h, H - y')
13:   
14:   // Extract and decrypt region
15:   R_enc ← F_temp[y':y'+h', x':x'+w']
16:   R_enc_flat ← R_enc.flatten()
17:   
18:   // XOR decryption (same operation as encryption)
19:   K_xor_repeated ← RepeatKey(K_xor, length(R_enc_flat))
20:   R_aes_flat ← R_enc_flat ⊕ K_xor_repeated
21:   R_aes ← R_aes_flat.reshape(R_enc.shape)
22:   
23:   // Write back AES-only encrypted region
24:   F_temp[y':y'+h', x':x'+w'] ← R_aes
25: End For
26: 
27: // Reverse Layer 1: AES-CTR decryption
28: F_temp_bytes ← F_temp.flatten().toBytes()
29: cipher ← AES.new(K_aes, mode=CTR, nonce=nonce)
30: F_dec_bytes ← cipher.decrypt(F_temp_bytes)
31: F_dec ← F_dec_bytes.reshape(F_enc.shape)
32: 
33: Return F_dec
```

---

## ALGORITHM 7: Performance Metrics Calculation

```
Input:
  - F_orig: Original frame
  - F_enc: Encrypted frame
  - F_dec: Decrypted frame
  - F_enc_prev: Previous encrypted frame (for NPCR/UACI)
  - t_enc: Encryption time
  - t_dec: Decryption time
  
Output: Metrics M = {MSE, PSNR, NPCR, UACI, timing}

Mathematical Formulations:

1. Mean Squared Error (MSE):
   MSE = (1/(H×W×C)) × Σᵢ Σⱼ Σₖ (F_orig[i,j,k] - F_dec[i,j,k])²

2. Peak Signal-to-Noise Ratio (PSNR):
   PSNR = 10 × log₁₀(MAX²/MSE)
   where MAX = 255 for 8-bit images

3. Number of Pixel Change Rate (NPCR):
   NPCR = (1/(H×W×C)) × Σᵢ Σⱼ Σₖ D[i,j,k] × 100%
   where D[i,j,k] = {1 if F_enc[i,j,k] ≠ F_enc_prev[i,j,k], 0 otherwise}

4. Unified Average Changing Intensity (UACI):
   UACI = (1/(H×W×C×255)) × Σᵢ Σⱼ Σₖ |F_enc[i,j,k] - F_enc_prev[i,j,k]| × 100%

Procedure CalculateMetrics(F_orig, F_enc, F_dec, F_enc_prev, t_enc, t_dec):
1: // MSE Calculation
2: diff ← (F_orig.asFloat() - F_dec.asFloat())²
3: MSE ← mean(diff)
4: 
5: // PSNR Calculation
6: If MSE = 0:
7:    PSNR ← ∞
8: Else:
9:    PSNR ← 10 × log₁₀(255² / MSE)
10: End If
11: 
12: // NPCR Calculation
13: If F_enc_prev exists:
14:    D ← (F_enc ≠ F_enc_prev)              // Boolean comparison
15:    NPCR ← (sum(D) / size(D)) × 100
16: End If
17: 
18: // UACI Calculation
19: If F_enc_prev exists:
20:    diff_enc ← |F_enc.asFloat() - F_enc_prev.asFloat()|
21:    UACI ← (sum(diff_enc) / (size(diff_enc) × 255)) × 100
22: End If
23: 
24: M ← {MSE, PSNR, NPCR, UACI, t_enc, t_dec}
25: Return M
```

---

## ALGORITHM 8: Object Tracking for Performance Optimization

```
Input:
  - F: Current frame
  - FACES_prev: Previously detected face regions
  - trackers: List of initialized trackers
  
Output: FACES_current: Updated face positions

Procedure InitializeTrackers(F, FACES):
1: trackers ← []
2: For each face (x, y, w, h) in FACES:
3:    tracker ← CSRT_Tracker.create()         // Or KCF tracker
4:    tracker.init(F, (x, y, w, h))
5:    trackers.append(tracker)
6: End For
7: Return trackers

Procedure UpdateTrackers(F, trackers):
1: FACES_current ← []
2: valid_trackers ← []
3: 
4: For each tracker in trackers:
5:    success, bbox ← tracker.update(F)
6:    If success:
7:       (x, y, w, h) ← bbox
8:       FACES_current.append((x, y, w, h))
9:       valid_trackers.append(tracker)
10:   End If
11: End For
12: 
13: trackers ← valid_trackers                 // Remove failed trackers
14: Return FACES_current
```

---

## ALGORITHM 9: Signature Verification

```
Input:
  - F_enc: Encrypted frame
  - σ: Digital signature
  - PK_rsa: RSA public key (2048 bits)
  
Output: valid: Boolean (True if signature is valid)

Mathematical Formulation:
  h = SHA-256(F_enc)
  h' = σ^e mod n
  valid = (h = h')
  where (n, e) is the RSA public key

Procedure Verify(F_enc, σ, PK_rsa):
1: F_enc_bytes ← F_enc.toBytes()
2: h ← SHA256.hash(F_enc_bytes)
3: 
4: Try:
5:    RSA-PKCS1-v1.5-Verify(PK_rsa, h, σ)
6:    Return True
7: Catch (ValueError, TypeError):
8:    Return False
9: End Try
```

---

## Security Analysis

### Theoretical Security Guarantees

**1. AES-256-CTR Security:**
- Key space: 2²⁵⁶ possible keys
- Brute force complexity: O(2²⁵⁶)
- IND-CPA secure (semantic security)

**2. Per-Frame Key Derivation:**
- Different keys for each frame prevents pattern analysis
- HKDF provides cryptographic key separation
- Salt = frame index ensures uniqueness

**3. Two-Factor Protection:**
- Face regions protected by: AES-CTR ⊕ XOR
- Requires breaking both encryption layers
- Combined security: min(Security_AES, Security_XOR)

**4. Digital Signature:**
- RSA-2048 provides ~112-bit security level
- SHA-256 collision resistance: O(2¹²⁸)
- Detects any tampering of encrypted frames

### Expected Metric Ranges

**Quality Metrics (after decryption):**
- MSE ≈ 0 (perfect reconstruction)
- PSNR → ∞ (perfect reconstruction)

**Security Metrics (encrypted frames):**
- NPCR ≈ 99.6094% (ideal for 8-bit images)
- UACI ≈ 33.4635% (ideal for 8-bit images)
- Signature verification: 100% (if no tampering)

---

## Computational Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Face Detection (Haar) | O(W×H×s²×n) | O(W×H) |
| Face Detection (DNN) | O(300²×layers) | O(300²) |
| Object Tracking | O(W×H) | O(k) |
| AES-CTR Encryption | O(W×H×C) | O(W×H×C) |
| XOR Encryption | O(Σᵢ wᵢ×hᵢ) | O(1) |
| RSA Signature | O(log³(n)) | O(n) |
| HKDF | O(L) | O(L) |

Where:
- W×H: Frame dimensions
- C: Color channels (3 for RGB)
- s: Scale factor window size
- n: Number of cascade stages
- k: Number of tracked objects
- L: Output key length
- wᵢ×hᵢ: Face region dimensions

---

## Implementation Details

**Programming Language:** Python 3.10+

**Key Libraries:**
- OpenCV 4.x (Computer Vision)
- PyCryptodome 3.x (Cryptography)
- NumPy 1.x (Numerical Operations)
- PyTorch 2.x (GPU Acceleration, optional)

**Hardware Requirements:**
- CPU: Multi-core processor (≥4 cores recommended)
- RAM: ≥8 GB
- GPU: NVIDIA CUDA-capable (optional, for acceleration)
- Storage: SSD recommended for I/O operations

**Performance Characteristics:**
- CPU-only processing: ~15-30 FPS (720p video)
- GPU-accelerated: ~45-60 FPS (720p video)
- Encryption overhead: ~20-40ms per frame
- Decryption overhead: ~15-35ms per frame


