import streamlit as st
import pandas as pd
from io import BytesIO

def compare_and_assign_status(tds_gl_full_df, revenue_register_full_df, tolerance=500):
    # Renaming columns for consistency within this function's scope
    # Also, renaming 'Document Header Text' to 'Reference_TDS' to make it unique
    tds_gl_full_df.rename(columns={'Amount in local currency': 'Amount', 'Document Header Text': 'Reference_TDS'}, inplace=True)
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
        ref_tds = tds_row['Reference_TDS']
        amount = round(tds_row['Amount'])

        # Attempt to match by Reference_TDS in the aggregated revenue summary
        matched_rows = aggregated_revenue[aggregated_revenue['Reference'] == ref_tds]
        if not matched_rows.empty:
            agg_amount = matched_rows['Amount'].iloc[0]  # Aggregated amount for the invoice
            if abs(agg_amount - amount) <= tolerance:
                match_status = 'Matched Directly/Matched By Sum'
            else:
                # If the aggregated amount doesn't match, check individual rows for a closer match
                individual_matched_rows = revenue_register_full_df[revenue_register_full_df['Reference'] == ref_tds]
                match_status = 'Amount Mismatch, Invoice Matched'  # Default status if no closer match found
                for rev_idx in individual_matched_rows.index:
                    rev_row = revenue_register_full_df.loc[rev_idx]
                    if abs(rev_row['Amount'] - amount) <= tolerance:
                        match_status = 'Matched Directly/Matched By Sum'  # Update if a closer match is found
                        break

            # Update TDS GL DataFrame match status and customer details from the first matched row
            first_matched_row = revenue_register_full_df[revenue_register_full_df['Reference'] == ref_tds].iloc[0]
            tds_gl_full_df.at[idx, 'Match Status'] = match_status
            tds_gl_full_df.at[idx, 'Cust Code'] = first_matched_row['Cust Code']
            tds_gl_full_df.at[idx, 'Customer Name'] = first_matched_row['Customer Name']
            # Update Revenue Register DataFrame match status for all matched rows
            revenue_register_full_df.loc[revenue_register_full_df['Reference'] == ref_tds, 'Match Status'] = match_status
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
    tds_gl_full_df.rename(columns={'Amount': 'Amount in local currency', 'Reference_TDS': 'Document Header Text'}, inplace=True)
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



def to_excel(df, sheet_name='Sheet1'):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        # Remove the writer.save() line
    output.seek(0)  # Go to the start of the stream
    return output.getvalue()

def main():
    st.title('TDS GL and Revenue Register Merger Utility')
    st.write("Download Blank template and fill Revenue Register")
    
    # Define the column headers based on the extracted columns
    column_headers = [
        'Invoice No.', 'Group/Non group', 'Revenue type', 'Date', 'Cust Code', 
        'Customer Name', 'GL CODE', 'Grouping', 'Qty - MT or Tonnes', 'Rate', 
        'Tax Category', 'Tax Rate', 'Basic Value (Rs.)', 'SGST (Rs.)', 'CGST (Rs.)', 
        'IGST (Rs.)', 'Invoice value (Rs.)', 'GSTIN', 'SAP Doc No', 'SAC CODE', 
        'HSN CODE', 'IRN', 'Date.1', 'Ack No', 'BARCODE', 
        'BOOKING STATUS/\nPAYMENT STATUS', 'TDS/TCS', 'NET RECEIVABLE'
    ]

    # Create an empty DataFrame with these headers
    df_blank = pd.DataFrame(columns=column_headers)

    # Convert DataFrame to Excel file in memory
    excel_file = to_excel(df_blank)
    
    # Create a download button in the Streamlit app to download the Excel file
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

        # Pass the user-defined tolerance to the function
        processed_tds_gl_df, processed_revenue_register_df = compare_and_assign_status(tds_gl_full_df, revenue_register_full_df, tolerance)

        # Converting DataFrames to CSV for download
        tds_gl_excel = to_excel(processed_tds_gl_df, 'TDS GL Data')
        revenue_register_excel = to_excel(processed_revenue_register_df, 'Revenue Register Data')
        
        summary_report_df = generate_summary_report(processed_tds_gl_df, processed_revenue_register_df)
        
        tds_pivot_table = create_pivot_table(
            processed_tds_gl_df,
            index=['Customer Name'],  # or the correct field name for the DataFrame index
            columns=['Match Status'],  # this should create headers from the Match Status values
            values='Amount in local currency'  # or the correct field name for the values
        )

        revenue_pivot_table = create_pivot_table(
            processed_revenue_register_df,
            index=['Customer Name'],  # same as above
            columns=['Match Status'],  # this should create headers from the Match Status values
            values='TDS/TCS'  # or the correct field name for the values
        )

        # Convert pivot tables to CSV for download
        tds_pivot_excel = to_excel(tds_pivot_table, 'TDS GL Pivot')
        revenue_pivot_excel = to_excel(revenue_pivot_table, 'Revenue Register Pivot')
        


        # Display pivot tables in Streamlit
        st.write("## TDS GL Pivot Table")
        st.dataframe(tds_pivot_table)
        st.write("## Revenue Register Pivot Table")
        st.dataframe(revenue_pivot_table)
        st.download_button("Download TDS GL Pivot Table",  tds_pivot_excel, "tds_gl_pivot.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key='download-tds-pivot')
        st.download_button("Download Revenue Register Pivot Table", revenue_pivot_excel, "revenue_register_pivot.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key='download-revenue-pivot')   
        st.divider()

# Display summary report
        st.write("## Summary Report")
        st.table(summary_report_df)
        st.divider()
        st.download_button("Download Processed TDS GL File", tds_gl_excel, "processed_tds_gl.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key='download-tds-gl')
        st.download_button("Download Processed Revenue Register File", revenue_register_excel, "processed_revenue_register.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key='download-revenue-register')

if __name__ == '__main__':
    main()
