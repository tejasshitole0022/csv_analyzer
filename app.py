from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from scipy import stats
import io
import base64
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Custom JSON encoder to handle NaN values
class NanSafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
        return super().default(obj)

app = Flask(__name__)
app.json_encoder = NanSafeJSONEncoder
app.secret_key = 'csv-analyzer-btech-project-2026'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def get_df():
    """Get current dataframe from session"""
    if 'csv_path' in session and os.path.exists(session['csv_path']):
        return pd.read_csv(session['csv_path'])
    return None

def save_df(df):
    """Save dataframe to session"""
    path = os.path.join(app.config['UPLOAD_FOLDER'], f'data_{session.get("user_id", "default")}.csv')
    df.to_csv(path, index=False)
    session['csv_path'] = path

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img_base64

@app.route('/')
def index():
    """Home page"""
    if 'user_id' not in session:
        session['user_id'] = datetime.now().strftime('%Y%m%d%H%M%S')
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Handle CSV file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            save_df(df)
            return jsonify({
                'success': True,
                'rows': len(df),
                'columns': len(df.columns),
                'message': 'File uploaded successfully'
            })
        
        return jsonify({'error': 'Invalid file type. Please upload CSV file'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/overview')
def overview():
    """Get dataset overview"""
    try:
        df = get_df()
        if df is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].to_dict()
        
        # Column details with data types and sample values
        column_details = []
        for col in df.columns:
            sample_vals = df[col].dropna().head(3).tolist()
            # Convert any NaN or inf to None for JSON serialization
            sample_vals = [str(v) if pd.notna(v) else None for v in sample_vals]
            
            col_info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'non_null': int(df[col].count()),
                'null': int(df[col].isnull().sum()),
                'unique': int(df[col].nunique()),
                'sample_values': sample_vals
            }
            
            # Add statistics for numeric columns
            if col in numeric_cols:
                try:
                    col_info['min'] = float(df[col].min()) if pd.notna(df[col].min()) else None
                    col_info['max'] = float(df[col].max()) if pd.notna(df[col].max()) else None
                    col_info['mean'] = float(df[col].mean()) if pd.notna(df[col].mean()) else None
                    col_info['median'] = float(df[col].median()) if pd.notna(df[col].median()) else None
                except:
                    pass
            
            column_details.append(col_info)
        
        # Convert head data, replacing NaN with None
        head_data = df.head(20).replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict('records')
        
        return jsonify({
            'rows': int(len(df)),
            'columns': int(len(df.columns)),
            'column_names': df.columns.tolist(),
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'dtypes': df.dtypes.astype(str).to_dict(),
            'head': head_data,
            'missing': missing_data,
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            'duplicates': int(df.duplicated().sum()),
            'column_details': column_details,
            'total_missing': int(df.isnull().sum().sum()),
            'missing_percentage': float((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100) if len(df) > 0 else 0
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics', methods=['POST'])
def statistics():
    """Get statistical summary"""
    try:
        df = get_df()
        if df is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        data = request.json
        columns = data.get('columns', df.select_dtypes(include=[np.number]).columns.tolist())
        
        stats_dict = {}
        for col in columns:
            if col in df.columns:
                if df[col].dtype in [np.float64, np.int64]:
                    col_data = df[col]
                    stats_dict[col] = {
                        'count': int(col_data.count()),
                        'mean': None if pd.isna(col_data.mean()) else float(col_data.mean()),
                        'std': None if pd.isna(col_data.std()) else float(col_data.std()),
                        'min': None if pd.isna(col_data.min()) else float(col_data.min()),
                        '25%': None if pd.isna(col_data.quantile(0.25)) else float(col_data.quantile(0.25)),
                        '50%': None if pd.isna(col_data.median()) else float(col_data.median()),
                        '75%': None if pd.isna(col_data.quantile(0.75)) else float(col_data.quantile(0.75)),
                        'max': None if pd.isna(col_data.max()) else float(col_data.max()),
                        'skewness': None if pd.isna(col_data.skew()) else float(col_data.skew()),
                        'kurtosis': None if pd.isna(col_data.kurtosis()) else float(col_data.kurtosis())
                    }
        
        return jsonify(stats_dict)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/correlation')
def correlation():
    """Generate correlation matrix"""
    try:
        df = get_df()
        if df is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        method = request.args.get('method', 'pearson')
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return jsonify({'error': 'Need at least 2 numeric columns'}), 400
        
        corr = df[numeric_cols].corr(method=method)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0, 
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title(f'Correlation Matrix ({method.capitalize()})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        img = fig_to_base64(fig)
        
        return jsonify({
            'image': img,
            'data': corr.to_dict(),
            'columns': numeric_cols
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/plot', methods=['POST'])
def plot():
    """Generate various plots"""
    try:
        df = get_df()
        if df is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        data = request.json
        plot_type = data.get('type')
        
        if plot_type == 'histogram':
            col = data.get('column')
            bins = data.get('bins', 30)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df[col].dropna(), bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'Histogram of {col}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
        elif plot_type == 'boxplot':
            col = data.get('column')
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bp = ax.boxplot(df[col].dropna(), vert=True, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            ax.set_ylabel(col, fontsize=12)
            ax.set_title(f'Box Plot of {col}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
        elif plot_type == 'scatter':
            x = data.get('x')
            y = data.get('y')
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df[x], df[y], alpha=0.6, c='steelblue', edgecolors='black')
            ax.set_xlabel(x, fontsize=12)
            ax.set_ylabel(y, fontsize=12)
            ax.set_title(f'Scatter Plot: {x} vs {y}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
        elif plot_type == 'bar':
            col = data.get('column')
            top_n = data.get('top_n', 10)
            
            value_counts = df[col].value_counts().head(top_n)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            value_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
            ax.set_xlabel(col, fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title(f'Top {top_n} Values in {col}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right')
            
        elif plot_type == 'line':
            x = data.get('x')
            y = data.get('y')
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df[x], df[y], marker='o', linestyle='-', linewidth=2, markersize=4)
            ax.set_xlabel(x, fontsize=12)
            ax.set_ylabel(y, fontsize=12)
            ax.set_title(f'Line Plot: {x} vs {y}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
        elif plot_type == 'distribution':
            col = data.get('column')
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Histogram with KDE
            axes[0].hist(df[col].dropna(), bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
            df[col].dropna().plot(kind='kde', ax=axes[0], color='red', linewidth=2)
            axes[0].set_xlabel(col, fontsize=12)
            axes[0].set_ylabel('Density', fontsize=12)
            axes[0].set_title('Distribution with KDE', fontsize=12, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # Q-Q Plot
            stats.probplot(df[col].dropna(), dist="norm", plot=axes[1])
            axes[1].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
        
        img = fig_to_base64(fig)
        return jsonify({'image': img})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clean/drop_columns', methods=['POST'])
def drop_columns():
    """Drop selected columns"""
    try:
        df = get_df()
        if df is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        columns = request.json.get('columns', [])
        df = df.drop(columns=columns)
        save_df(df)
        
        return jsonify({'success': True, 'message': f'Dropped {len(columns)} columns'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clean/fill_missing', methods=['POST'])
def fill_missing():
    """Fill missing values"""
    try:
        df = get_df()
        if df is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        data = request.json
        col = data.get('column')
        method = data.get('method')
        
        if method == 'mean':
            df[col].fillna(df[col].mean(), inplace=True)
        elif method == 'median':
            df[col].fillna(df[col].median(), inplace=True)
        elif method == 'mode':
            df[col].fillna(df[col].mode()[0], inplace=True)
        elif method == 'forward':
            df[col].fillna(method='ffill', inplace=True)
        elif method == 'backward':
            df[col].fillna(method='bfill', inplace=True)
        elif method == 'custom':
            value = data.get('value')
            df[col].fillna(value, inplace=True)
        elif method == 'drop':
            df.dropna(subset=[col], inplace=True)
        
        save_df(df)
        return jsonify({'success': True, 'message': f'Filled missing values in {col}'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clean/remove_duplicates', methods=['POST'])
def remove_duplicates():
    """Remove duplicate rows"""
    try:
        df = get_df()
        if df is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        data = request.json or {}
        columns = data.get('columns', [])
        
        before = len(df)
        
        if columns and len(columns) > 0:
            # Remove duplicates based on specific columns
            df = df.drop_duplicates(subset=columns)
            message = f'Removed {before - len(df)} duplicate rows based on columns: {", ".join(columns)}'
        else:
            # Remove duplicates based on all columns
            df = df.drop_duplicates()
            message = f'Removed {before - len(df)} duplicate rows'
        
        after = len(df)
        
        save_df(df)
        return jsonify({
            'success': True,
            'removed': before - after,
            'message': message
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clean/remove_outliers', methods=['POST'])
def remove_outliers():
    """Remove outliers using IQR method"""
    try:
        df = get_df()
        if df is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        column = request.json.get('column')
        
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        before = len(df)
        df = df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]
        after = len(df)
        
        save_df(df)
        return jsonify({
            'success': True,
            'removed': before - after,
            'message': f'Removed {before - after} outliers from {column}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/pca', methods=['POST'])
def pca_analysis():
    """Perform PCA analysis"""
    return jsonify({'error': 'Feature removed'}), 404

@app.route('/api/clustering', methods=['POST'])
def clustering():
    """Perform K-Means clustering"""
    return jsonify({'error': 'Feature removed'}), 404

@app.route('/api/ml/regression', methods=['POST'])
def regression():
    """Perform regression analysis"""
    return jsonify({'error': 'Feature removed'}), 404

@app.route('/api/ml/classification', methods=['POST'])
def classification():
    """Perform classification analysis"""
    return jsonify({'error': 'Feature removed'}), 404

@app.route('/api/download')
def download():
    """Download current dataset"""
    try:
        df = get_df()
        if df is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        output = io.BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'cleaned_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset to original uploaded data"""
    try:
        if 'csv_path' in session:
            os.remove(session['csv_path'])
            session.pop('csv_path', None)
        return jsonify({'success': True, 'message': 'Session reset'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("CSV DATA ANALYZER - B.Tech Project")
    print("=" * 60)
    print("Server starting on http://localhost:5000")
    print("Press CTRL+C to quit")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
