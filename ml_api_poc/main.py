import io
import pickle

import pandas as pd
from fastapi import FastAPI, UploadFile

# from app.utils import utils

app = FastAPI()
def preporcessing(Test_Beneficiarydata, Test_Inpatientdata, Test_Outpatientdata):
    # Test_Beneficiarydata.head(200).to_csv('Test_Beneficiarydata.csv')
    # Test_Inpatientdata.head(300).to_csv('Test_Inpatientdata.csv')
    # Test_Outpatientdata.head(300).to_csv('Test_Outpatientdata.csv')
    ##Replacing 2 with 0 for chronic conditions ,that means chroniv condition No is 0 and yes is 1

    # Train_Beneficiarydata = Train_Beneficiarydata.replace({'ChronicCond_Alzheimer': 2, 'ChronicCond_Heartfailure': 2, 'ChronicCond_KidneyDisease': 2,
    #                            'ChronicCond_Cancer': 2, 'ChronicCond_ObstrPulmonary': 2, 'ChronicCond_Depression': 2,
    #                            'ChronicCond_Diabetes': 2, 'ChronicCond_IschemicHeart': 2, 'ChronicCond_Osteoporasis': 2,
    #                            'ChronicCond_rheumatoidarthritis': 2, 'ChronicCond_stroke': 2 }, 0)

    # Train_Beneficiarydata = Train_Beneficiarydata.replace({'RenalDiseaseIndicator': 'Y'}, 1)

    Test_Beneficiarydata = Test_Beneficiarydata.replace({'ChronicCond_Alzheimer': 2, 'ChronicCond_Heartfailure': 2, 'ChronicCond_KidneyDisease': 2,
                            'ChronicCond_Cancer': 2, 'ChronicCond_ObstrPulmonary': 2, 'ChronicCond_Depression': 2,
                            'ChronicCond_Diabetes': 2, 'ChronicCond_IschemicHeart': 2, 'ChronicCond_Osteoporasis': 2,
                            'ChronicCond_rheumatoidarthritis': 2, 'ChronicCond_stroke': 2 }, 0)

    Test_Beneficiarydata = Test_Beneficiarydata.replace({'RenalDiseaseIndicator': 'Y'}, 1)

    Test_Beneficiarydata['DOB'] = pd.to_datetime(Test_Beneficiarydata['DOB'], format='%Y-%m-%d')
    Test_Beneficiarydata['DOD'] = pd.to_datetime(Test_Beneficiarydata['DOD'], format='%Y-%m-%d', errors='ignore')
    Test_Beneficiarydata['Age'] = round(((Test_Beneficiarydata['DOD'] - Test_Beneficiarydata['DOB']).dt.days) / 365)

    ## As we see that last DOD value is 2009-12-01 ,which means Beneficiary Details data is of year 2009.
    ## so we will calculate age of other benficiaries for year 2009
    Test_Beneficiarydata.Age.fillna(
        round(((pd.to_datetime('2009-12-01', format='%Y-%m-%d') - Test_Beneficiarydata['DOB']).dt.days) / 365),
        inplace=True)

    # Lets create a new variable 'WhetherDead' with flag 1 means Dead and 0 means not Dead
    Test_Beneficiarydata.loc[Test_Beneficiarydata.DOD.isna(), 'WhetherDead'] = 0
    Test_Beneficiarydata.loc[Test_Beneficiarydata.DOD.notna(), 'WhetherDead'] = 1

    ## As patient can be admitted for only for 1 day,we will add 1 to the difference of Discharge Date and Admission Date


    Test_Inpatientdata['AdmissionDt'] = pd.to_datetime(Test_Inpatientdata['AdmissionDt'], format='%Y-%m-%d')
    Test_Inpatientdata['DischargeDt'] = pd.to_datetime(Test_Inpatientdata['DischargeDt'], format='%Y-%m-%d')
    Test_Inpatientdata['AdmitForDays'] = ((Test_Inpatientdata['DischargeDt'] - Test_Inpatientdata[
        'AdmissionDt']).dt.days) + 1

    # **Merge Inpatient and Outpatinet data and create dataset for all patients.
    # **

    # In[10]:

    # Lets make union of Inpatienta and outpatient data .
    # We will use all keys in outpatient data as we want to make union and dont want duplicate columns from both tables.

    Test_Allpatientdata = pd.merge(Test_Outpatientdata, Test_Inpatientdata,
                                   left_on=['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider',
                                            'InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician',
                                            'OtherPhysician', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2',
                                            'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5',
                                            'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
                                            'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10', 'ClmProcedureCode_1',
                                            'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4',
                                            'ClmProcedureCode_5', 'ClmProcedureCode_6', 'DeductibleAmtPaid',
                                            'ClmAdmitDiagnosisCode'],
                                   right_on=['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider',
                                             'InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician',
                                             'OtherPhysician', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2',
                                             'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4', 'ClmDiagnosisCode_5',
                                             'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
                                             'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10', 'ClmProcedureCode_1',
                                             'ClmProcedureCode_2', 'ClmProcedureCode_3', 'ClmProcedureCode_4',
                                             'ClmProcedureCode_5', 'ClmProcedureCode_6', 'DeductibleAmtPaid',
                                             'ClmAdmitDiagnosisCode']
                                   , how='outer')

    # Lets merge All patient data with beneficiary details data based on 'BeneID' as joining key for inner join
    Test_AllPatientDetailsdata = pd.merge(Test_Allpatientdata, Test_Beneficiarydata, left_on='BeneID',
                                          right_on='BeneID', how='inner')

    print('Shape of All Patient Details Test : ', Test_AllPatientDetailsdata.shape)

    # Lets merge patient data with fradulent providers details data with "Provider" as joining key for inner join

    # Test_ProviderWithPatientDetailsdata = pd.merge(Test, Test_AllPatientDetailsdata, on='Provider')
    Test_ProviderWithPatientDetailsdata = Test_AllPatientDetailsdata

    Test_ProviderWithPatientDetailsdata["PerProviderAvg_InscClaimAmtReimbursed"] = \
    Test_ProviderWithPatientDetailsdata.groupby('Provider')['InscClaimAmtReimbursed'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerProviderAvg_DeductibleAmtPaid"] = \
    Test_ProviderWithPatientDetailsdata.groupby('Provider')['DeductibleAmtPaid'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerProviderAvg_IPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('Provider')['IPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerProviderAvg_IPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('Provider')['IPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerProviderAvg_OPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('Provider')['OPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerProviderAvg_OPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('Provider')['OPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerProviderAvg_Age"] = Test_ProviderWithPatientDetailsdata.groupby('Provider')[
        'Age'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerProviderAvg_NoOfMonths_PartACov"] = \
    Test_ProviderWithPatientDetailsdata.groupby('Provider')['NoOfMonths_PartACov'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerProviderAvg_NoOfMonths_PartBCov"] = \
    Test_ProviderWithPatientDetailsdata.groupby('Provider')['NoOfMonths_PartBCov'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerProviderAvg_AdmitForDays"] = \
    Test_ProviderWithPatientDetailsdata.groupby('Provider')['AdmitForDays'].transform('mean')

    ## Grouping based on BeneID explains amounts involved per beneficiary.Reason to derive this feature is that one beneficiary
    ## can go to multiple providers and can be involved in fraud cases


    Test_ProviderWithPatientDetailsdata["PerBeneIDAvg_InscClaimAmtReimbursed"] = \
    Test_ProviderWithPatientDetailsdata.groupby('BeneID')['InscClaimAmtReimbursed'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerBeneIDAvg_DeductibleAmtPaid"] = \
    Test_ProviderWithPatientDetailsdata.groupby('BeneID')['DeductibleAmtPaid'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerBeneIDAvg_IPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('BeneID')['IPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerBeneIDAvg_IPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('BeneID')['IPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerBeneIDAvg_OPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('BeneID')['OPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerBeneIDAvg_OPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('BeneID')['OPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerBeneIDAvg_AdmitForDays"] = \
    Test_ProviderWithPatientDetailsdata.groupby('BeneID')['AdmitForDays'].transform('mean')

    ### Average features grouped by OtherPhysician.


    Test_ProviderWithPatientDetailsdata["PerOtherPhysicianAvg_InscClaimAmtReimbursed"] = \
    Test_ProviderWithPatientDetailsdata.groupby('OtherPhysician')['InscClaimAmtReimbursed'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerOtherPhysicianAvg_DeductibleAmtPaid"] = \
    Test_ProviderWithPatientDetailsdata.groupby('OtherPhysician')['DeductibleAmtPaid'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerOtherPhysicianAvg_IPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('OtherPhysician')['IPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerOtherPhysicianAvg_IPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('OtherPhysician')['IPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerOtherPhysicianAvg_OPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('OtherPhysician')['OPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerOtherPhysicianAvg_OPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('OtherPhysician')['OPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerOtherPhysicianAvg_AdmitForDays"]=Test_ProviderWithPatientDetailsdata.groupby('OtherPhysician')['AdmitForDays'].transform('mean')

    ##Average features grouped by OperatingPhysician

    Test_ProviderWithPatientDetailsdata["PerOperatingPhysicianAvg_InscClaimAmtReimbursed"] = \
    Test_ProviderWithPatientDetailsdata.groupby('OperatingPhysician')['InscClaimAmtReimbursed'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerOperatingPhysicianAvg_DeductibleAmtPaid"] = \
    Test_ProviderWithPatientDetailsdata.groupby('OperatingPhysician')['DeductibleAmtPaid'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerOperatingPhysicianAvg_IPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('OperatingPhysician')['IPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerOperatingPhysicianAvg_IPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('OperatingPhysician')['IPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerOperatingPhysicianAvg_OPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('OperatingPhysician')['OPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerOperatingPhysicianAvg_OPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('OperatingPhysician')['OPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerOperatingPhysicianAvg_AdmitForDays"] = \
    Test_ProviderWithPatientDetailsdata.groupby('OperatingPhysician')['AdmitForDays'].transform('mean')

    # **Average features grouped by AttendingPhysician**

    # In[22]:

    ### Average features grouped by AttendingPhysician

    Test_ProviderWithPatientDetailsdata["PerAttendingPhysicianAvg_InscClaimAmtReimbursed"] = \
    Test_ProviderWithPatientDetailsdata.groupby('AttendingPhysician')['InscClaimAmtReimbursed'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerAttendingPhysicianAvg_DeductibleAmtPaid"] = \
    Test_ProviderWithPatientDetailsdata.groupby('AttendingPhysician')['DeductibleAmtPaid'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerAttendingPhysicianAvg_IPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('AttendingPhysician')['IPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerAttendingPhysicianAvg_IPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('AttendingPhysician')['IPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerAttendingPhysicianAvg_OPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('AttendingPhysician')['OPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerAttendingPhysicianAvg_OPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('AttendingPhysician')['OPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerAttendingPhysicianAvg_AdmitForDays"] = \
    Test_ProviderWithPatientDetailsdata.groupby('AttendingPhysician')['AdmitForDays'].transform('mean')

    # **Average features grouped by DiagnosisGroupCode**

    # In[23]:

    ###  Average features grouped by DiagnosisGroupCode

    Test_ProviderWithPatientDetailsdata["PerDiagnosisGroupCodeAvg_InscClaimAmtReimbursed"] = \
    Test_ProviderWithPatientDetailsdata.groupby('DiagnosisGroupCode')['InscClaimAmtReimbursed'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerDiagnosisGroupCodeAvg_DeductibleAmtPaid"] = \
    Test_ProviderWithPatientDetailsdata.groupby('DiagnosisGroupCode')['DeductibleAmtPaid'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerDiagnosisGroupCodeAvg_IPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('DiagnosisGroupCode')['IPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerDiagnosisGroupCodeAvg_IPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('DiagnosisGroupCode')['IPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerDiagnosisGroupCodeAvg_OPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('DiagnosisGroupCode')['OPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerDiagnosisGroupCodeAvg_OPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('DiagnosisGroupCode')['OPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerDiagnosisGroupCodeAvg_AdmitForDays"] = \
    Test_ProviderWithPatientDetailsdata.groupby('DiagnosisGroupCode')['AdmitForDays'].transform('mean')

    # **Average features grouped by ClmAdmitDiagnosisCode**

    # In[24]:

    ### Average features grouped by ClmAdmitDiagnosisCode

    Test_ProviderWithPatientDetailsdata["PerClmAdmitDiagnosisCodeAvg_InscClaimAmtReimbursed"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmAdmitDiagnosisCode')['InscClaimAmtReimbursed'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmAdmitDiagnosisCodeAvg_DeductibleAmtPaid"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmAdmitDiagnosisCode')['DeductibleAmtPaid'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmAdmitDiagnosisCodeAvg_IPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmAdmitDiagnosisCode')['IPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmAdmitDiagnosisCodeAvg_IPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmAdmitDiagnosisCode')['IPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmAdmitDiagnosisCodeAvg_OPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmAdmitDiagnosisCode')['OPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmAdmitDiagnosisCodeAvg_OPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmAdmitDiagnosisCode')['OPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmAdmitDiagnosisCodeAvg_AdmitForDays"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmAdmitDiagnosisCode')['AdmitForDays'].transform('mean')

    # **Average features grouped by ClmProcedureCode_1**

    # In[25]:

    ### Average features grouped by ClmProcedureCode_1


    Test_ProviderWithPatientDetailsdata["PerClmProcedureCode_1Avg_InscClaimAmtReimbursed"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmProcedureCode_1')['InscClaimAmtReimbursed'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmProcedureCode_1Avg_DeductibleAmtPaid"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmProcedureCode_1')['DeductibleAmtPaid'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmProcedureCode_1Avg_IPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmProcedureCode_1')['IPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmProcedureCode_1Avg_IPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmProcedureCode_1')['IPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmProcedureCode_1Avg_OPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmProcedureCode_1')['OPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmProcedureCode_1Avg_OPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmProcedureCode_1')['OPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmProcedureCode_1Avg_AdmitForDays"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmProcedureCode_1')['AdmitForDays'].transform('mean')

    # **Average features grouped by ClmProcedureCode_2**

    # In[26]:

    ### Average features grouped by ClmProcedureCode_2


    Test_ProviderWithPatientDetailsdata["PerClmProcedureCode_2Avg_InscClaimAmtReimbursed"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmProcedureCode_2')['InscClaimAmtReimbursed'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmProcedureCode_2Avg_DeductibleAmtPaid"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmProcedureCode_2')['DeductibleAmtPaid'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmProcedureCode_2Avg_IPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmProcedureCode_2')['IPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmProcedureCode_2Avg_IPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmProcedureCode_2')['IPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmProcedureCode_2Avg_OPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmProcedureCode_2')['OPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmProcedureCode_2Avg_OPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmProcedureCode_2')['OPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmProcedureCode_2Avg_AdmitForDays"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmProcedureCode_2')['AdmitForDays'].transform('mean')

    # **Average features grouped by ClmProcedureCode_3**

    # In[27]:

    ###  Average features grouped by ClmProcedureCode_3


    Test_ProviderWithPatientDetailsdata["PerClmProcedureCode_3Avg_InscClaimAmtReimbursed"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmProcedureCode_3')['InscClaimAmtReimbursed'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmProcedureCode_3Avg_DeductibleAmtPaid"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmProcedureCode_3')['DeductibleAmtPaid'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmProcedureCode_3Avg_IPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmProcedureCode_3')['IPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmProcedureCode_3Avg_IPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmProcedureCode_3')['IPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmProcedureCode_3Avg_OPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmProcedureCode_3')['OPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmProcedureCode_3Avg_OPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmProcedureCode_3')['OPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmProcedureCode_3Avg_AdmitForDays"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmProcedureCode_3')['AdmitForDays'].transform('mean')

    # **Average features grouped by ClmDiagnosisCode_1**

    # In[28]:

    ### Average features grouped by ClmDiagnosisCode_1


    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_1Avg_InscClaimAmtReimbursed"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_1')['InscClaimAmtReimbursed'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_1Avg_DeductibleAmtPaid"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_1')['DeductibleAmtPaid'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_1Avg_IPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_1')['IPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_1Avg_IPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_1')['IPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_1Avg_OPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_1')['OPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_1Avg_OPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_1')['OPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_1Avg_AdmitForDays"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_1')['AdmitForDays'].transform('mean')

    # **Average features grouped by ClmDiagnosisCode_2**

    # In[29]:

    ###  Average features grouped by ClmDiagnosisCode_2


    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_2Avg_InscClaimAmtReimbursed"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_2')['InscClaimAmtReimbursed'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_2Avg_DeductibleAmtPaid"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_2')['DeductibleAmtPaid'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_2Avg_IPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_2')['IPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_2Avg_IPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_2')['IPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_2Avg_OPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_2')['OPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_2Avg_OPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_2')['OPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_2Avg_AdmitForDays"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_2')['AdmitForDays'].transform('mean')

    # **Average features grouped by ClmDiagnosisCode_3
    # **

    # In[30]:

    ###  Average features grouped by ClmDiagnosisCode_3


    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_3Avg_InscClaimAmtReimbursed"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_3')['InscClaimAmtReimbursed'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_3Avg_DeductibleAmtPaid"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_3')['DeductibleAmtPaid'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_3Avg_IPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_3')['IPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_3Avg_IPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_3')['IPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_3Avg_OPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_3')['OPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_3Avg_OPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_3')['OPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_3Avg_AdmitForDays"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_3')['AdmitForDays'].transform('mean')

    # **Average features grouped by ClmDiagnosisCode_4**

    # In[31]:

    ###  Average features grouped by ClmDiagnosisCode_4



    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_4Avg_InscClaimAmtReimbursed"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_4')['InscClaimAmtReimbursed'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_4Avg_DeductibleAmtPaid"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_4')['DeductibleAmtPaid'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_4Avg_IPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_4')['IPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_4Avg_IPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_4')['IPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_4Avg_OPAnnualReimbursementAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_4')['OPAnnualReimbursementAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_4Avg_OPAnnualDeductibleAmt"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_4')['OPAnnualDeductibleAmt'].transform('mean')
    Test_ProviderWithPatientDetailsdata["PerClmDiagnosisCode_4Avg_AdmitForDays"] = \
    Test_ProviderWithPatientDetailsdata.groupby('ClmDiagnosisCode_4')['AdmitForDays'].transform('mean')

    # **Claims are filed by Provider,so fraud can be organized crime.So we will check ClmCounts filed by Providers and when pairs like Provider +BeneID, Provider+Attending Physician, Provider+ClmAdmitDiagnosisCode, Provider+ClmProcedureCode_1,Provider+ClmDiagnosisCode_1 are together.**
    #
    #
    # **Average Feature based on grouping based on combinations of different variables.**

    # In[32]:

    ### Average Feature based on grouping based on combinations of different variables


    Test_ProviderWithPatientDetailsdata["ClmCount_Provider"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider'])['ClaimID'].transform('count')
    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_BeneID"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'BeneID'])['ClaimID'].transform('count')
    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_AttendingPhysician"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'AttendingPhysician'])['ClaimID'].transform('count')
    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_OtherPhysician"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'OtherPhysician'])['ClaimID'].transform('count')
    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_OperatingPhysician"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'OperatingPhysician'])['ClaimID'].transform('count')
    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_ClmAdmitDiagnosisCode"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'ClmAdmitDiagnosisCode'])['ClaimID'].transform('count')
    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_ClmProcedureCode_1"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'ClmProcedureCode_1'])['ClaimID'].transform('count')
    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_ClmProcedureCode_2"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'ClmProcedureCode_2'])['ClaimID'].transform('count')
    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_ClmProcedureCode_3"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'ClmProcedureCode_3'])['ClaimID'].transform('count')
    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_ClmProcedureCode_4"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'ClmProcedureCode_4'])['ClaimID'].transform('count')
    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_ClmProcedureCode_5"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'ClmProcedureCode_5'])['ClaimID'].transform('count')
    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_ClmDiagnosisCode_1"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'ClmDiagnosisCode_1'])['ClaimID'].transform('count')
    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_ClmDiagnosisCode_2"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'ClmDiagnosisCode_2'])['ClaimID'].transform('count')
    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_ClmDiagnosisCode_3"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'ClmDiagnosisCode_3'])['ClaimID'].transform('count')
    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_ClmDiagnosisCode_4"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'ClmDiagnosisCode_4'])['ClaimID'].transform('count')
    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_ClmDiagnosisCode_5"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'ClmDiagnosisCode_5'])['ClaimID'].transform('count')
    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_ClmDiagnosisCode_6"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'ClmDiagnosisCode_6'])['ClaimID'].transform('count')
    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_ClmDiagnosisCode_7"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'ClmDiagnosisCode_7'])['ClaimID'].transform('count')
    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_ClmDiagnosisCode_8"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'ClmDiagnosisCode_8'])['ClaimID'].transform('count')
    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_ClmDiagnosisCode_9"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'ClmDiagnosisCode_9'])['ClaimID'].transform('count')
    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_DiagnosisGroupCode"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'DiagnosisGroupCode'])['ClaimID'].transform('count')

    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_BeneID_AttendingPhysician"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'BeneID', 'AttendingPhysician'])['ClaimID'].transform(
        'count')
    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_BeneID_OtherPhysician"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'BeneID', 'OtherPhysician'])['ClaimID'].transform('count')
    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_BeneID_AttendingPhysician_ClmProcedureCode_1"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'BeneID', 'AttendingPhysician', 'ClmProcedureCode_1'])[
        'ClaimID'].transform('count')
    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_BeneID_AttendingPhysician_ClmDiagnosisCode_1"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'BeneID', 'AttendingPhysician', 'ClmDiagnosisCode_1'])[
        'ClaimID'].transform('count')
    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_BeneID_OperatingPhysician"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'BeneID', 'OperatingPhysician'])['ClaimID'].transform(
        'count')
    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_BeneID_ClmProcedureCode_1"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'BeneID', 'ClmProcedureCode_1'])['ClaimID'].transform(
        'count')
    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_BeneID_ClmDiagnosisCode_1"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'BeneID', 'ClmDiagnosisCode_1'])['ClaimID'].transform(
        'count')
    Test_ProviderWithPatientDetailsdata["ClmCount_Provider_BeneID_ClmDiagnosisCode_1_ClmProcedureCode_1"] = \
    Test_ProviderWithPatientDetailsdata.groupby(['Provider', 'BeneID', 'ClmDiagnosisCode_1', 'ClmProcedureCode_1'])[
        'ClaimID'].transform('count')

    ## Lets Check unique values of ICD Diagnosis Codes


    # **Feature Selection**

    # In[35]:

    # Lets remove unnecessary columns ,as we grouped based on these columns and derived maximum infromation from them.


    remove_these_columns = ['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'AttendingPhysician',
                            'OperatingPhysician', 'OtherPhysician', 'ClmDiagnosisCode_1',
                            'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
                            'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7',
                            'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9', 'ClmDiagnosisCode_10',
                            'ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3',
                            'ClmProcedureCode_4', 'ClmProcedureCode_5', 'ClmProcedureCode_6',
                            'ClmAdmitDiagnosisCode', 'AdmissionDt',
                            'DischargeDt', 'DiagnosisGroupCode', 'DOB', 'DOD',
                            'State', 'County']

    Test_category_removed = Test_ProviderWithPatientDetailsdata.drop(axis=1, columns=remove_these_columns)

    # **Type Conversion**

    # In[36]:

    ## Lets Convert types of gender and race to categorical.

    Test_category_removed.Gender = Test_category_removed.Gender.astype('category')
    Test_category_removed.Race = Test_category_removed.Race.astype('category')

    # **Dummification**

    # In[37]:

    # Lets create dummies for categorrical columns.

    Test_category_removed = pd.get_dummies(Test_category_removed, columns=['Gender', 'Race'], drop_first=True)

    # **Convert Target values to 1 and 0,wher '1' means Yes and '0' means No**


    # In[39]:

    # Test_category_removed = Test_category_removed.iloc[:135392]  ##Remove train data from appended test data

    # **Data Aggregation to the Providers level**

    Test_category_removed[['RenalDiseaseIndicator','Gender_2' ,'Race_2','Race_3','Race_5' ]]=Test_category_removed[['RenalDiseaseIndicator','Gender_2' ,'Race_2','Race_3','Race_5' ]].astype('int')

    # In[40]:

    ### Lets aggregate claims data to unique providers.
    Test_category_removed_groupedbyProv_PF = Test_category_removed.groupby(['Provider'], as_index=False).agg('sum')
    print(Test_category_removed_groupedbyProv_PF.shape)
    provider_id=Test_category_removed_groupedbyProv_PF['Provider']
    # In[41]:

    ## Lets Seperate out Target and providers from independent variables.Create Target column y.


    # **Standardization**

    # In[42]:

    ## Lets apply StandardScaler and transform values to its z form,where 99.7% values range between -3 to 3.
    # sc = StandardScaler()  # MinMaxScaler


    # X_teststd = sc.transform(
    #     Test_category_removed_groupedbyProv_PF.iloc[:, 1:])  # Apply Standard Scaler to unseen data as well.



    #########################################################################################################################

    # refrence: appliedroots.com

    def predict_with_best_t(proba, threshould):
        predictions = []
        for i in proba:
            if i >= threshould:
                predictions.append(1)
            else:
                predictions.append(0)
        return predictions

    # loading the best model
    print('model_loaded')

    # X_train['RenalDiseaseIndicator'] = X_train['RenalDiseaseIndicator'].astype(str).astype(int)
    filename = 'app/models/best_rf_model_pkl.sav'
    scaler= 'app/models/scaler.pkl'

    loaded_model = pickle.load(open(filename, 'rb'))

    X_scaled_unseen = pickle.load(open(scaler, 'rb'))
    # Test_category_removed_groupedbyProv_PF['RenalDiseaseIndicator']=0
    unseen= X_scaled_unseen.transform(Test_category_removed_groupedbyProv_PF.iloc[:, 1:])
    X_teststd = loaded_model.predict(unseen)

    # y_predict_tr = model.predict_proba(X_train)[:, 1]
    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.predict(X_teststd)

    # prediction_tr = predict_with_best_t(y_predict_tr, 0.382)

    pred_df = pd.DataFrame(X_teststd, columns=['Prediction'])


    return pred_df, provider_id

@app.get("/home")
def read_root():
    return {"message": "This is home"}



@app.post("/predict/")
async def predict(beneficiary: UploadFile, inpatient: UploadFile, outpatient: UploadFile):
    if (
            beneficiary.filename.endswith(".csv")
            and inpatient.filename.endswith(".csv")
            and outpatient.filename.endswith(".csv")
    ):
        beneficiary_contents = await beneficiary.read()
        inpatient_contents = await inpatient.read()
        outpatient_contents = await outpatient.read()

        beneficiary_df = pd.read_csv(io.BytesIO(beneficiary_contents))
        inpatient_df = pd.read_csv(io.BytesIO(inpatient_contents))
        outpatient_df = pd.read_csv(io.BytesIO(outpatient_contents))

        df, provider_id = preporcessing(beneficiary_df, inpatient_df, outpatient_df)
        result_df = pd.concat([df, provider_id], axis=1)

        df['Prediction'] = df['Prediction'].map({0: 'No', 1: 'Yes'})

        return result_df[['Prediction', 'Provider']].to_dict(orient='records')

