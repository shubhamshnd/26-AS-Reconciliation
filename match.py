import streamlit as st
import pandas as pd
from io import BytesIO
from fuzzywuzzy import process

def compare_and_assign_status(tds_gl_full_df, revenue_register_full_df, tolerance=500):
    # Renaming columns for consistency within this function's scope
    tds_gl_full_df.rename(columns={'Amount in local currency': 'Amount'}, inplace=True)
    revenue_register_full_df.rename(columns={'Invoice No.': 'Reference', 'TDS/TCS': 'Amount'}, inplace=True)

    # Check if 'Cust Code', 'Customer Name', and 'Match Status' columns exist in TDS GL DataFrame, if not, initialize them
    if 'Cust Code' not in tds_gl_full_df.columns:
        tds_gl_full_df['Cust Code'] = None
    if 'Customer Name' not in tds_gl_full_df.columns:
        tds_gl_full_df['Customer Name'] = None
    if 'Match Status' not in tds_gl_full_df.columns:
        tds_gl_full_df['Match Status'] = 'Not matched'
    
    # Ensure 'Match Status' column exists in Revenue Register DataFrame
    if 'Match Status' not in revenue_register_full_df.columns:
        revenue_register_full_df['Match Status'] = 'Not matched'

    # Aggregating TDS/TCS values by Invoice No. in the revenue register
    aggregated_revenue = revenue_register_full_df.groupby('Reference').agg({'Amount': 'sum'}).reset_index()

    for idx, tds_row in tds_gl_full_df.iterrows():
        ref = tds_row['Reference']
        amount = round(tds_row['Amount'])

        # Attempt to match by reference first in the aggregated revenue summary
        matched_rows = aggregated_revenue[aggregated_revenue['Reference'] == ref]
        if not matched_rows.empty:
            agg_amount = matched_rows['Amount'].iloc[0]  # Aggregated amount for the invoice
            if abs(agg_amount - amount) <= tolerance:
                match_status = 'Matched Directly/Matched By Sum'
            else:
                # If the aggregated amount doesn't match, check individual rows for a closer match
                individual_matched_rows = revenue_register_full_df[revenue_register_full_df['Reference'] == ref]
                match_status = 'Amount Mismatch, Invoice Matched'  # Default status if no closer match found
                for rev_idx in individual_matched_rows.index:
                    rev_row = revenue_register_full_df.loc[rev_idx]
                    if abs(rev_row['Amount'] - amount) <= tolerance:
                        match_status = 'Matched Directly/Matched By Sum'  # Update if a closer match is found
                        break

            # Update TDS GL DataFrame match status and customer details from the first matched row
            first_matched_row = revenue_register_full_df[revenue_register_full_df['Reference'] == ref].iloc[0]
            tds_gl_full_df.at[idx, 'Match Status'] = match_status
            tds_gl_full_df.at[idx, 'Cust Code'] = first_matched_row['Cust Code']
            tds_gl_full_df.at[idx, 'Customer Name'] = first_matched_row['Customer Name']
            # Update Revenue Register DataFrame match status for all matched rows
            revenue_register_full_df.loc[revenue_register_full_df['Reference'] == ref, 'Match Status'] = match_status
        else:
            # If no direct reference match, attempt to match by amount in the original dataframe
            matched_by_amount = revenue_register_full_df[revenue_register_full_df['Amount'].round() == amount]
            if not matched_by_amount.empty:
                for rev_idx in matched_by_amount.index:
                    rev_row = matched_by_amount.loc[rev_idx]
                    if revenue_register_full_df.at[rev_idx, 'Match Status'] == 'Not matched':
                        match_status = 'Invoice Mismatch, Amount Matched'
                        tds_gl_full_df.at[idx, 'Match Status'] = match_status
                        tds_gl_full_df.at[idx, 'Cust Code'] = rev_row['Cust Code']
                        tds_gl_full_df.at[idx, 'Customer Name'] = rev_row['Customer Name']
                        revenue_register_full_df.at[rev_idx, 'Match Status'] = match_status

    # Reorder columns in TDS GL DataFrame
    tds_cols_order = ['Match Status', 'Cust Code', 'Customer Name'] + [col for col in tds_gl_full_df.columns if col not in ['Match Status', 'Cust Code', 'Customer Name']]
    tds_gl_full_df = tds_gl_full_df[tds_cols_order]

    # Reorder columns in Revenue Register DataFrame to have 'Match Status' first
    rev_cols_order = ['Match Status'] + [col for col in revenue_register_full_df.columns if col != 'Match Status']
    revenue_register_full_df = revenue_register_full_df[rev_cols_order]

    # Reset column names to their original form, if needed, for consistency
    tds_gl_full_df.rename(columns={'Amount': 'Amount in local currency'}, inplace=True)
    revenue_register_full_df.rename(columns={'Reference': 'Invoice No.', 'Amount': 'TDS/TCS'}, inplace=True)

    return tds_gl_full_df, revenue_register_full_df


def generate_summary_report(tds_gl_df, revenue_register_df):
    # Summary counts for different match statuses
    summary = {
        'Total Records in TDS GL': len(tds_gl_df),
        'Total Records in Revenue Register': len(revenue_register_df),
        'Matched by Sum': (tds_gl_df['Match Status'] == 'Matched Directly/Matched By Sum').sum(),
        'Invoice Mismatch, Amount Matched': (tds_gl_df['Match Status'] == 'Invoice Mismatch, Amount Matched').sum(),
        'Amount Mismatch, Invoice Matched': (tds_gl_df['Match Status'] == 'Amount Mismatch, Invoice Matched').sum(),
        'Not matched': (tds_gl_df['Match Status'] == 'Not matched').sum(),
    }

    # Convert summary to DataFrame for easy display in Streamlit
    summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Count'])
    return summary_df

def create_pivot_table(df, index, columns, values):
    df[index] = df[index].fillna('Blank')
    df[values] = pd.to_numeric(df[values], errors='coerce')

    # Ensure that 'Match Status' is of type string to prevent any aggregation
    df['Match Status'] = df['Match Status'].astype(str)
    
    # Create the pivot table
    pivot_table = pd.pivot_table(
        df,
        index=index,              # This is the leftmost part of the pivot table
        columns=columns,          # These are the new headers
        values=values,            # These are the values to be summarized
        aggfunc='sum',            # This is the aggregation function to combine the values
        fill_value=0,             # Filling missing values with zero
        margins=True,             # Adds the grand total
        margins_name='Grand Total'  # Name of the grand total row
    )
    
    # Reorder the levels if Match Status is a MultiIndex
    if isinstance(pivot_table.columns, pd.MultiIndex):
        pivot_table = pivot_table.reorder_levels([1, 0], axis=1).sort_index(axis=1)

    return pivot_table


def parse_26as_text_file(file_content):
    normalized_content = file_content.replace('^', '\t')
    lines = normalized_content.split("\n")
    data = []  # List to hold all rows for the dataframe

    # Detect delimiter by checking the first non-empty line

    current_deductor_name = ""  # Variable to hold the deductor's name
    current_deductor_tan = ""  # Variable to hold the deductor's TAN
    
    # Attempt to extract header information, ensuring we don't cause an index error
    if len(lines) > 3:
        header_info_line = lines[3].strip()  # Assuming the 4th line contains the header values
        header_info = header_info_line.split('\t')
        # Adjusting for the actual number of header columns based on the structure
        header_info_adjusted = header_info[:13]  # Adjust to match the number of header columns
    else:
        # Default to empty if the expected header isn't present
        header_info_adjusted = ["N/A"] * 13  # Adjust based on the expected number of header columns

    # Define the header columns based on the user's description
    header_columns = [
        'File Creation Date', 'Permanent Account Number (PAN)', 'Current Status of PAN', 
        'Financial Year', 'Assessment Year', 'Name of Assessee', 
        'Address Line 1', 'Address Line 2', 'Address Line 3', 
        'Address Line 4', 'Address Line 5', 'Statecode', 'Pin Code'
    ]

    # Process each line in the file for transaction and deductor summary
    for line in lines:
        line = line.strip()

        # Skip empty lines and header lines for transaction details
        if not line or ('Sr. No.' in line and 'Section' in line):
            continue

        parts = line.split('\t')

        # Check if this line is a deductor summary line
        if len(parts) > 7 and parts[2] and parts[6] == '':
            current_deductor_name = parts[1]
            current_deductor_tan = parts[2]
        # Check if this line is a transaction detail line
        elif len(parts) > 8 and parts[0].isdigit():
            # Append the current deductor details and transaction details, including header info
            transaction_data = [current_deductor_name, current_deductor_tan] + parts[1:9] + header_info_adjusted
            data.append(transaction_data)

    # Define the dataframe with the correct columns, including the new header info columns
    columns = [
        'Name of Deductor', 'TAN of Deductor', 'Section', 'Transaction Date',
        'Status of Booking', 'Date of Booking', 'Remarks',
        'Amount Paid / Credited(Rs.)', 'Tax Deducted(Rs.)', 'TDS Deposited(Rs.)'
    ] + header_columns
    
    df = pd.DataFrame(data, columns=columns)
    
    return df



def match_records(df_26AS, df_tds_gl, error_margin=50):
    df_26AS['Match Status'] = 'Not Reconciled'
    df_26AS['Cust Code'] = ''
    df_26AS['Reference'] = ''
    df_26AS['Document Number'] = ''
    df_26AS['Amount in local currency (TDS GL)'] = 0.0

    for index, row_26AS in df_26AS.iterrows():
        best_match = process.extractOne(row_26AS['Name of Deductor'], df_tds_gl['Customer Name'].unique(), score_cutoff=80)
        if best_match:
            matched_records = df_tds_gl[df_tds_gl['Customer Name'] == best_match[0]]
            for _, row_tds_gl in matched_records.iterrows():
                if abs(round(row_26AS['TDS Deposited(Rs.)']) - round(row_tds_gl['Amount in local currency'])) <= error_margin:
                    df_26AS.at[index, 'Match Status'] = 'Reconciled'
                    df_26AS.at[index, 'Cust Code'] = row_tds_gl['Cust Code']
                    df_26AS.at[index, 'Reference'] = row_tds_gl['Reference']
                    df_26AS.at[index, 'Document Number'] = row_tds_gl['Document Number']
                    df_26AS.at[index, 'Amount in local currency (TDS GL)'] = row_tds_gl['Amount in local currency']
                    break

    cols = ['Match Status', 'Cust Code', 'Reference', 'Document Number', 'Amount in local currency (TDS GL)'] + [col for col in df_26AS.columns if col not in ['Match Status', 'Cust Code', 'Reference', 'Document Number', 'Amount in local currency (TDS GL)']]
    return df_26AS[cols]

def load_data(file):
    return pd.read_excel(file)


def to_excel(df, sheet_name='Sheet1'):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        # Remove the writer.save() line
    output.seek(0)  # Go to the start of the stream
    return output.getvalue()


def show_readme():
    # Function to display README contents
    with open("README.md", "r", encoding="utf-8") as file:
        readme_contents = file.read()
    st.markdown(readme_contents)

def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    st.sidebar.header("Instructions")
    st.sidebar.write("This app provides utilities for TDS GL and Revenue Register Merger, "
                     "26AS to Excel Conversion, and Reconciliation. Select a section from "
                     "the radio buttons below to proceed with the specific utility. "
                     "Ensure to have all necessary files prepared and formatted according "
                     "to the requirements mentioned in each section.")
    
    if 'show_readme' not in st.session_state:
        st.session_state.show_readme = False
    
    if st.sidebar.button("Show README") and not st.session_state.show_readme:
        st.session_state.show_readme = True
    if st.session_state.show_readme:
        show_readme()
        if st.button("Back to App"):
            st.session_state.show_readme = False
        return  # Early return to avoid displaying the rest of the app
    
    section = st.sidebar.radio("Select a Section", ['Section 1 - TDS GL and Revenue Register Merger', 
                                                    'Section 2 - 26AS to Excel Conversion', 
                                                    'Section 3 - Reconciliation'])
    

    
    
    if section == 'Section 1 - TDS GL and Revenue Register Merger':
        st.title('TDS GL and Revenue Register Merger Utility')
        st.write("Download the Blank template and fill in the Revenue Register")

        column_headers = [
            'Invoice No.', 'Group/Non group', 'Revenue type', 'Date', 'Cust Code', 
            'Customer Name', 'GL CODE', 'Grouping', 'Qty - MT or Tonnes', 'Rate', 
            'Tax Category', 'Tax Rate', 'Basic Value (Rs.)', 'SGST (Rs.)', 'CGST (Rs.)', 
            'IGST (Rs.)', 'Invoice value (Rs.)', 'GSTIN', 'SAP Doc No', 'SAC CODE', 
            'HSN CODE', 'IRN', 'Date.1', 'Ack No', 'BARCODE', 
            'BOOKING STATUS/\nPAYMENT STATUS', 'TDS/TCS', 'NET RECEIVABLE'
        ]

        df_blank = pd.DataFrame(columns=column_headers)
        excel_file = to_excel(df_blank)
        
        st.download_button(
            label="Download Blank Revenue Register Template",
            data=excel_file,
            file_name="Revenue_Register_Format.xlsx",
            mime="application/vnd.ms-excel"
        )
        st.divider()
        tolerance = st.number_input('ENTER MAXIMUM TOLERANCE FOR MATCHING', value=500, step=1)
        st.divider()
        tds_gl_file = st.file_uploader("UPLOAD SAP EXPORTED TDS GL FILE", type=['xlsx'])
        revenue_register_file = st.file_uploader("UPLOAD REVENUE REGISTER WITH ADDED DATA", type=['xlsx'])
        st.divider()
        if tds_gl_file and revenue_register_file:
            tds_gl_full_df = pd.read_excel(tds_gl_file)
            revenue_register_full_df = pd.read_excel(revenue_register_file)
            processed_tds_gl_df, processed_revenue_register_df = compare_and_assign_status(tds_gl_full_df, revenue_register_full_df, tolerance)
            tds_gl_excel = to_excel(processed_tds_gl_df, 'TDS GL Data')
            revenue_register_excel = to_excel(processed_revenue_register_df, 'Revenue Register Data')
            summary_report_df = generate_summary_report(processed_tds_gl_df, processed_revenue_register_df)
            tds_pivot_table = create_pivot_table(processed_tds_gl_df, index=['Customer Name'], columns=['Match Status'], values='Amount in local currency')
            revenue_pivot_table = create_pivot_table(processed_revenue_register_df, index=['Customer Name'], columns=['Match Status'], values='TDS/TCS')
            tds_pivot_excel = to_excel(tds_pivot_table, 'TDS GL Pivot')
            revenue_pivot_excel = to_excel(revenue_pivot_table, 'Revenue Register Pivot')
            st.write("## TDS GL Pivot Table")
            st.dataframe(tds_pivot_table)
            st.write("## Revenue Register Pivot Table")
            st.dataframe(revenue_pivot_table)
            st.download_button("Download TDS GL Pivot Table",  tds_pivot_excel, "tds_gl_pivot.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key='download-tds-pivot')
            st.download_button("Download Revenue Register Pivot Table", revenue_pivot_excel, "revenue_register_pivot.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key='download-revenue-pivot')   
            st.divider()
            st.write("## Summary Report")
            st.table(summary_report_df)
            st.divider()
            st.download_button("Download Processed TDS GL File", tds_gl_excel, "processed_tds_gl.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key='download-tds-gl')
            st.download_button("Download Processed Revenue Register File", revenue_register_excel, "processed_revenue_register.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key='download-revenue-register')

    elif section == 'Section 2 - 26AS to Excel Conversion':
        st.title("26AS Text File Processor")
        st.write("## Upload 26AS text file from government portal")
        text_file_26as = st.file_uploader("Upload 26AS File (Text)", type=['txt'])
        if text_file_26as:
            file_content = text_file_26as.getvalue().decode("utf-8")
            as26_df = parse_26as_text_file(file_content)
            st.write("Preview of Processed 26AS Data:")
            st.dataframe(as26_df)
            st.download_button(
                label="Download Excel",
                data=to_excel(as26_df),
                file_name="26AS_Data.xlsx",
                mime="application/vnd.ms-excel"
            )
        
    elif section == 'Section 3 - Reconciliation':
        st.title("26AS Reconciliation Tool")
        st.write("## Upload Previously exported 26AS and TDS GL Excel Files")
        file_26AS = st.file_uploader("Upload 26AS file", type=['xlsx'])
        file_tds_gl = st.file_uploader("Upload TDS GL file", type=['xlsx'])
        if file_26AS and file_tds_gl:
            data_26AS = pd.read_excel(file_26AS)
            data_tds_gl = pd.read_excel(file_tds_gl)
            if st.button('Match Records'):
                matched_data = match_records(data_26AS, data_tds_gl)
                st.write("## Matched Records")
                st.dataframe(matched_data)

                total_rows_26AS = len(data_26AS)
                reconciled = matched_data['Match Status'].value_counts().get('Reconciled', 0)
                not_reconciled = total_rows_26AS - reconciled
                reconciliation_percentage = (reconciled / total_rows_26AS) * 100 if total_rows_26AS else 0
                
                st.write("## Match Report")
                st.write(f"- Total Rows in 26AS File: {total_rows_26AS}")
                st.write(f"- Reconciled: {reconciled}")
                st.write(f"- Not Reconciled: {not_reconciled}")
                st.write(f"- Reconciliation Percentage: {reconciliation_percentage:.2f}%")

                excel_data = to_excel(matched_data, 'Matched Records')
                st.download_button(label='Download Matched Data as Excel',
                                   data=excel_data,
                                   file_name='matched_26AS_data.xlsx',
                                   mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                
                
if __name__ == '__main__':
    main()
