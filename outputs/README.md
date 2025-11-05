# Output Files Organization

This folder contains all generated outputs from the Video Encryption System.

## 📁 Folder Structure

```
outputs/
├── csv/                    # CSV data files
│   ├── metrics_per_frame.csv       # Detailed per-frame encryption metrics
│   ├── metrics_summary.csv         # Summary statistics
│   └── video_size_comparison.csv   # File size comparison data
│
├── images/                 # Generated charts and visualizations
│   ├── algorithm_comparison.png
│   ├── detailed_comparison.png
│   ├── metrics_histograms.png
│   ├── metrics_timeline.png
│   └── quality_metrics_detailed.png
│
└── reports/                # HTML reports
    └── video_size_comparison.html  # Interactive comparison report
```

## 📊 File Descriptions

### CSV Files (`outputs/csv/`)

**metrics_per_frame.csv**
- Per-frame encryption/decryption times
- MSE, PSNR, NPCR, UACI values
- Face detection counts
- Signature verification status

**metrics_summary.csv**
- Overall processing statistics
- Average encryption/decryption times
- File size information
- Security metrics averages

**video_size_comparison.csv**
- Original, encrypted, and decrypted file sizes
- Processing time breakdown
- Performance metrics
- Quality metrics summary

### Images (`outputs/images/`)

**algorithm_comparison.png**
- Comparison of different encryption algorithms
- Performance vs security trade-offs

**detailed_comparison.png**
- Detailed performance analysis
- Time series data

**metrics_histograms.png**
- Distribution of encryption metrics
- Statistical analysis

**metrics_timeline.png**
- Performance over frame sequence
- Temporal analysis

**quality_metrics_detailed.png**
- PSNR, MSE, NPCR, UACI visualizations
- Quality assessment charts

### Reports (`outputs/reports/`)

**video_size_comparison.html**
- Interactive HTML report
- File size comparison
- Processing time analysis
- Quality metrics dashboard
- Open in web browser for best viewing

## 🔄 Regenerating Outputs

All files in this folder are automatically generated. To regenerate:

1. **Run encryption process:**
   ```bash
   python main.py
   ```
   Generates: `metrics_per_frame.csv`, `metrics_summary.csv`

2. **Generate comparison report:**
   ```bash
   python video_size_comparison.py
   ```
   Generates: `video_size_comparison.csv`, `video_size_comparison.html`

3. **Generate algorithm comparisons:**
   ```bash
   python algorithm_comparison_table.py
   ```
   (If script exists - generates algorithm comparison outputs)

## 📝 Notes

- CSV files can be opened in Excel, Google Sheets, or any spreadsheet software
- PNG images can be viewed in any image viewer
- HTML reports are best viewed in a modern web browser (Chrome, Firefox, Edge)
- All outputs are safe to delete - they will be regenerated on next run

