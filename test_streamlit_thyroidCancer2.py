"""
1. Error reading gene list file: expected str, bytes or os.PathLike object, not UploadedFile
2. gene_list_file 고정으로 하기!

"""
import streamlit as st
import pandas as pd
import numpy as np
from utils_tabnet import extractFeature  # readGenelist는 사용하지 않습니다.
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
from io import BytesIO

# Define device for PyTorch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# gene_list 파일 읽는 방식 수정
def read_genelist_fixed(file_path):
    """Read gene list from a fixed file path."""
    try:
        with open(file_path, 'r') as f:
            gene_list = f.read().splitlines()  # 줄 단위로 리스트 생성
        return gene_list
    except Exception as e:
        st.error(f"Error reading gene list file: {str(e)}")
        return None
    
def test(data, model_path, result_path):
    try:
        # Prepare data
        X_test = data.values

        # Define model
        model = TabNetClassifier()

        # Load weights
        model.load_model(model_path)

        # Test
        y_predprob = model.predict_proba(X_test)
        y_predvalue = model.predict(X_test)
        y_pred = np.hstack((y_predprob[:, -1].reshape(-1, 1), y_predvalue.reshape(-1, 1)))

        # Save result
        result = pd.DataFrame(y_pred, index=data.index, columns=['predict proba', 'predict value'])
        result['predict value'] = result['predict value'].apply(lambda x: 'N1' if x else 'N0')
        result.to_csv(result_path)
        return result
    except Exception as e:
        st.error(f"An error occurred during the model testing phase: {str(e)}")
        return None

# Function to convert DataFrame to CSV for download
def convert_df_to_csv(df):
    try:
        return df.to_csv().encode('utf-8')
    except Exception as e:
        st.error(f"Error converting DataFrame to CSV: {str(e)}")
        return None

# Function to convert DataFrame to Excel for download
def convert_df_to_excel(df):
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=True)
        processed_data = output.getvalue()
        return processed_data
    except Exception as e:
        st.error(f"Error converting DataFrame to Excel: {str(e)}")
        return None

# Streamlit interface
def main():
    st.title(" Thyroid Gene Classification ")

    #### 사용자메뉴얼 추가
    st.write("### 사용자 메뉴얼 다운로드")
    ## 메뉴얼 pdf파일 경로
    user_manual_path = './data/user_manual.pdf'
    try:
        # 메뉴얼 PDF 파일 열기
        with open(user_manual_path, "rb") as pdf_file:
            PDFbyte = pdf_file.read()
        
        # PDF 다운로드 버튼 생성
        st.download_button(
            label="Download User Manual (PDF)",
            data=PDFbyte,
            file_name="user_manual.pdf",
            mime='application/pdf'
        )
    except Exception as e:
        st.error(f"Error loading the user manual: {str(e)}")


    st.write("### Drag & Drop Data File (CSV or Excel)")

    # File uploads (Drag & Drop)
    data_file = st.file_uploader("Upload Data CSV or Excel (Drag & Drop)", type=["csv","xlsx"])
    #gene_list_file = st.file_uploader("Upload Gene List TXT (Drag & Drop)", type=["txt"])
    #gene_list_file = st.text_input("Feture Gene List","data/gene_f_classif_corr_top9.txt")
    # 고정된 gene_list 파일 경로
    gene_list_file = './data/gene_f_classif_corr_top9.txt'

    # Model weight file path
    model_path = st.text_input("Model Weight Path", 'weights/tabnet_f_classif_val0.87_test0.87.zip')
    
    # Result file path
    result_path = st.text_input("Result CSV Path", 'result.csv')

    # Additional parameters
    batch_size = st.number_input("Batch Size", min_value=1, max_value=100, value=20)
    epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=500)
    LR = st.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=0.01, step=0.0001)
    n_features = st.number_input("Number of Features", min_value=1, max_value=100, value=9)

    # Start processing when user clicks button
    if st.button("Run Test"):
        try:
            # Check if both files are uploaded
            if data_file is None:
                st.error("Please upload the data file (CSV or Excel).")
                return
            if gene_list_file is None:
                st.error("Please upload the gene list file.")
                return
            
            # Read data
            try:
                if data_file.name.endswith('.csv'):
                    data = pd.read_csv(data_file, index_col=0)
                elif data_file.name.endswith('.xlsx'):
                    data = pd.read_excel(data_file, index_col=0)
                else:
                    st.error("Unsupported file format. Please upload a CSV or Excel file.")
                    return
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                return

            # Read gene list from fixed file path
            gene_list = read_genelist_fixed(gene_list_file)
            if gene_list is None:
                return
            
            # # Read gene list
            # try:
            #     gene_list_content = gene_list_file.read().decode('utf-8')
            #     # 유전자 리스트를 줄바꿈 기준으로 분리하여 리스트로 변환
            #     gene_list = [gene.strip() for gene in gene_list_content.strip().split('\n') if gene.strip()]
            # except Exception as e:
            #     st.error(f"Error reading gene list file: {str(e)}")
            #     return

            # Preprocess data
            try:
                data = extractFeature(data, nor=True)
                data = data[gene_list]
            except Exception as e:
                st.error(f"Error during data preprocessing: {str(e)}")
                return

            # Run test and get results
            result = test(data, model_path, result_path)
            if result is not None:
                # Display results
                st.dataframe(result)

                # CSV download
                csv = convert_df_to_csv(result)
                if csv is not None:
                    st.download_button(label="Download as CSV", data=csv, file_name='result.csv', mime='text/csv')

                # Excel download
                excel = convert_df_to_excel(result)
                if excel is not None:
                    st.download_button(label="Download as Excel", data=excel, file_name='result.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                
                st.success(f"Results saved to {result_path}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
