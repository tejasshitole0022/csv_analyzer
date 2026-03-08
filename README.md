# CSV Data Analyzer - B.Tech Final Year Project

## 🎓 Project Overview
Advanced web-based CSV data analysis platform with statistical analysis, data cleaning, visualization, and PCA capabilities.

## ✨ Features

### 1. **Data Upload & Overview**
- Upload CSV files (up to 50MB)
- Comprehensive dataset overview
- Column-wise information (data types, missing values, unique counts)
- Memory usage statistics
- Duplicate detection

### 2. **Data Cleaning**
- Drop unwanted columns
- Fill missing values (Mean, Median, Mode, Forward/Backward Fill, Custom)
- Remove duplicate rows
- Remove outliers using IQR method

### 3. **Statistical Analysis**
- Descriptive statistics (mean, std, min, max, quartiles, skewness, kurtosis)
- Correlation matrix with multiple methods (Pearson, Kendall, Spearman)
- Interactive heatmap visualization

### 4. **Data Visualizations**
- Histogram with customizable bins
- Box plots
- Scatter plots
- Bar charts (top N values)
- Line plots
- Distribution analysis with KDE and Q-Q plots

### 5. **PCA (Principal Component Analysis)**
- Dimensionality reduction
- Variance explained analysis
- Scree plot
- 2D PCA projection visualization
- Component loadings

## 🛠️ Technology Stack

**Backend:**
- Flask (Python web framework)
- Pandas (Data manipulation)
- NumPy (Numerical computing)
- Scikit-learn (Machine learning & PCA)
- Matplotlib & Seaborn (Visualizations)
- SciPy (Statistical functions)

**Frontend:**
- HTML5
- CSS3 (Modern responsive design)
- JavaScript (ES6+)
- Font Awesome (Icons)

## 📋 Requirements

```
Flask==3.0.0
pandas==2.1.4
numpy==1.26.2
matplotlib==3.8.2
seaborn==0.13.0
scikit-learn==1.3.2
scipy==1.11.4
Werkzeug==3.0.1
```

## 🚀 Installation & Setup

### 1. Clone or Download the Project
```bash
cd CSV_Analyzer
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

### 5. Access the Application
Open your browser and navigate to:
```
http://localhost:5000
```

## 📖 Usage Guide

### Step 1: Upload CSV File
- Click "Choose File" button
- Select your CSV file
- Wait for upload confirmation

### Step 2: Explore Overview
- View dataset statistics
- Check column information
- Identify missing values and duplicates

### Step 3: Clean Data
- Remove unwanted columns
- Handle missing values
- Remove duplicates and outliers

### Step 4: Analyze Data
- Generate descriptive statistics
- Create correlation matrix
- Identify relationships between variables

### Step 5: Visualize Data
- Select plot type
- Choose columns
- Generate interactive visualizations

### Step 6: Apply PCA
- Configure number of components
- Enable/disable standardization
- Analyze variance explained
- View dimensionality reduction results

### Step 7: Download Cleaned Data
- Click "Download CSV" button
- Save processed dataset

## 📁 Project Structure

```
CSV_Analyzer/
├── app.py                  # Flask backend application
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html         # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css      # Styling
│   └── js/
│       └── app.js         # Frontend JavaScript
└── uploads/               # Temporary file storage
```

## 🎯 Key Functionalities

### Data Cleaning Operations
- **Drop Columns**: Remove unnecessary features
- **Fill Missing**: Multiple imputation strategies
- **Remove Duplicates**: Eliminate redundant rows
- **Remove Outliers**: IQR-based outlier detection

### Statistical Analysis
- **Descriptive Stats**: Complete statistical summary
- **Correlation Analysis**: Relationship identification
- **Distribution Analysis**: Normality testing

### Visualizations
- **Univariate**: Histograms, box plots, distributions
- **Bivariate**: Scatter plots, line plots
- **Categorical**: Bar charts with top N values

### Advanced Analytics
- **PCA**: Dimensionality reduction and feature extraction
- **Variance Analysis**: Component importance evaluation
- **Data Transformation**: Standardization and scaling

## 🔒 Security Features
- File size limit (50MB)
- CSV file type validation
- Session-based data isolation
- Secure file handling

## 🎨 UI/UX Features
- Modern gradient design
- Responsive layout
- Loading indicators
- Success/Error notifications
- Interactive tabs
- Smooth animations

## 📊 Sample Use Cases

1. **Exploratory Data Analysis**: Quick insights into dataset structure
2. **Data Preprocessing**: Clean and prepare data for modeling
3. **Feature Engineering**: Identify important features using PCA
4. **Statistical Reporting**: Generate comprehensive data reports
5. **Data Quality Assessment**: Identify and fix data issues

## 🐛 Troubleshooting

### Issue: Module not found
**Solution**: Install all requirements
```bash
pip install -r requirements.txt
```

### Issue: Port already in use
**Solution**: Change port in app.py
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Issue: File upload fails
**Solution**: Check file size (<50MB) and format (.csv)

## 📝 Future Enhancements
- Export analysis reports (PDF)
- More visualization types
- Time series analysis
- Advanced ML models
- Database integration
- User authentication

## 👨‍💻 Developer Information
**Project Type**: B.Tech Final Year Project  
**Domain**: Data Science & Web Development  
**Year**: 2026

## 📄 License
This project is created for educational purposes.

## 🙏 Acknowledgments
- Flask Documentation
- Scikit-learn Documentation
- Pandas Documentation
- Stack Overflow Community

---

**Note**: This is a B.Tech final year project demonstrating data analysis capabilities using modern web technologies.
