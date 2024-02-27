import pickle
from tkinter.filedialog import askopenfilename

import numpy as np
import pandas as pd


class DataController(object):

    @staticmethod
    def create_dataframe_from_DTA():
        """" reads the SPSS file and convert into a dataframe."""
        try:
            # dta file type with categorical data
            data = pd.read_stata('hse_2019_eul_20211006.dta', convert_categoricals=False)

            cols = data.columns.tolist()
            # non-unique categorical values to be removed
            cols.remove("ParSm")  # 4 2019
            cols.remove("CParSm")  # 4 2019
            cols.remove("WHOutc")  # 6 both

            data_frame = pd.read_stata('hse_2019_eul_20211006.dta', columns=cols)

            return data_frame

        except FileNotFoundError:
            raise FileNotFoundError
        except TypeError:
            raise TypeError
        except ValueError:
            raise ValueError

    @staticmethod
    def prep_data(data_frame):
        """" set the column names, index the IDs and remove constants and dup data"""
        try:
            # set labels back into the data frame
            data_frame = DataController.set_column_labels(data_frame)
            data_frame = pd.DataFrame(data=data_frame)

            # columns with missing data
            # print(data_frame.isnull().mean() * 100)

            # manual editing
            data_frame = DataController.column_cleaning(data_frame)

            # print(data_frame.columns[data_frame.isna().any()].tolist())
            # data_frame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

            # Drops constant value columns of dataframe.
            keep_columns = data_frame.columns[data_frame.nunique() > 1]
            data_frame = data_frame.loc[:, keep_columns]

            # Create correlation matrix.
            corr_matrix = data_frame.corr().abs()
            # Select upper triangle of correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
            # Find features with correlation greater than 0.90
            to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

            data_frame = data_frame.drop(to_drop, axis=1)
            data_frame = pd.DataFrame(data=data_frame)

            # create chart of class imbalance
            # value_counts = data_frame[ "(D) Any prescribed Antipsychotic medications taken in last 7 days (binary)"].value_counts()
            # value_counts.plot(kind="bar", title="Class distribution of the target variable")
            # plt.show()

            # format for modeling
            features = data_frame.columns
            dataframe = data_frame.values
            dataframe = pd.DataFrame(data=dataframe)

            # set x or data without target and y to target
            last_ix = len(dataframe.columns) - 1
            X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
            X = pd.DataFrame(data=X)
            X = X.infer_objects()
            y = y.infer_objects()

            return X, y, features

        except TypeError as te:
            raise TypeError(f"Must be a JSON file. {te}")
        except ValueError as ve:
            raise ValueError(f"File not in correct format. {ve}")

    @staticmethod
    def column_cleaning(data_frame):

        data_frame = DataController.columns_to_remove_by_file(data_frame)

        # set index
        data_frame.set_index("Archive respondent serial number", inplace=True)
        data_frame.drop(columns=['HSE 2019 Weight for analysis of cotinine sample (age 4+)'], inplace=True)

        data_frame = pd.DataFrame(data=data_frame)

        # Other degree or professional
        data_frame['Doctorate qualification'].replace("Not applicable", 'Not mentioned', inplace=True)
        data_frame['Masters qualification'].replace("Not applicable", 'Not mentioned', inplace=True)
        data_frame['Undergraduate or first degree qualification'].replace("Not applicable", 'Not mentioned',
                                                                          inplace=True)
        data_frame['Foundation degree'].replace("Not applicable", 'Not mentioned', inplace=True)
        data_frame['Graduate membership of a professional institution qualification'].replace("Not applicable",
                                                                                              'Not mentioned',
                                                                                              inplace=True)
        data_frame['Other post graduate degree or professional qualification'].replace("Not applicable",
                                                                                       'Not mentioned', inplace=True)

        # on phy/mental health
        data_frame['Whether had a fit/convulsion in last five years'] \
            .replace("Not applicable", 'No', inplace=True)

        data_frame['Food dominated your life during the last year'] \
            .replace("Not applicable", 'No', inplace=True)
        data_frame['Believe yourself to be fat when others said you were too thin during the last year'] \
            .replace("Not applicable", 'No', inplace=True)
        data_frame['Worry you had lost control over how much you eat during the last year'] \
            .replace("Not applicable", 'No', inplace=True)
        data_frame[
            'Feelings about food interfere with your ability to work, meet personal responsibilities and/or enjoy a social life during the last year'] \
            .replace("Not applicable", 'No', inplace=True)
        data_frame['Made yourself be sick because you felt uncomfortably full during the last year'] \
            .replace("Not applicable", 'No', inplace=True)
        data_frame['Lost more than one stone in a 3 month period during the last year'] \
            .replace("Not applicable", 'No', inplace=True)

        data_frame['Whether pays for any private care at the moment'] \
            .replace("Not applicable", 'No support needed', inplace=True)
        data_frame['Whether receives any care paid for by the council or local authority'] \
            .replace("Not applicable", 'No support needed', inplace=True)
        data_frame['Whether has Personal Budget'] \
            .replace("Not applicable", 'No support needed', inplace=True)
        data_frame['Whether council or local authority has made an assessment/review of care needs'] \
            .replace("Not applicable", 'No support needed', inplace=True)
        data_frame['Which of these things talked to doctor about'] \
            .replace("Not applicable", 'No GP', inplace=True)
        data_frame[
            'In the last 12 months, approximately how many times talked to, or visited a GP or family doctor about own health'] \
            .replace("Not applicable", 'No GP', inplace=True)
        data_frame['How long day-to-day activities have been reduced'] \
            .replace("Not applicable", 'No long term illness', inplace=True)
        data_frame['Day-to-day activities reduced due to illness'] \
            .replace("Not applicable", 'No long term illness', inplace=True)
        data_frame['Whether conditions or illnesses affect: None of these'] \
            .replace("Not applicable", 'No mental/physical problems', inplace=True)
        data_frame['Whether conditions or illnesses affect: Other'] \
            .replace("Not applicable", 'No mental/physical problems', inplace=True)
        data_frame[
            "Whether conditions or illnesses affect: Socially or behaviourally (for example associated with autism, attention deficit disorder or Asperger's syndrome)"] \
            .replace("Not applicable", 'No mental/physical problems', inplace=True)
        data_frame['Whether conditions or illnesses affect: Stamina, breathing or fatigue'] \
            .replace("Not applicable", 'No mental/physical problems', inplace=True)
        data_frame['Whether conditions or illnesses affect: Mental health'] \
            .replace("Not applicable", 'No mental/physical problems', inplace=True)
        data_frame['Whether conditions or illnesses affect: Memory'] \
            .replace("Not applicable", 'No mental/physical problems', inplace=True)
        data_frame['Whether conditions or illnesses affect: Learning or understanding or concentrating'] \
            .replace("Not applicable", 'No mental/physical problems', inplace=True)
        data_frame[
            'Whether conditions or illnesses affect: Dexterity (for example lifting and carrying objects, using a keyboard)'] \
            .replace("Not applicable", 'No mental/physical problems', inplace=True)
        data_frame[
            'Whether conditions or illnesses affect: Mobility (for example walking short distances or climbing stairs)'] \
            .replace("Not applicable", 'No mental/physical problems', inplace=True)
        data_frame['Whether conditions or illnesses affect: Hearing (for example deafness or partial hearing)'] \
            .replace("Not applicable", 'No mental/physical problems', inplace=True)
        data_frame['Whether conditions or illnesses affect: Vision (for example blindness or partial sight)'] \
            .replace("Not applicable", 'No mental/physical problems', inplace=True)

        # on income
        data_frame['(D) Joint household income - recoded'].replace("Not applicable", 'Not joint household',
                                                                   inplace=True)
        data_frame['Whether other income in household'].replace("Not applicable", 'No', inplace=True)

        # on Nicotine
        data_frame['Nicotine replacement product used to help stop smoking: Nicotine chewing gum (capi+casi)'].replace(
            "Not applicable", 'No',
            inplace=True)
        data_frame['Nicotine replacement product used to help stop smoking: Nicotine lozenges (capi+casi)'].replace(
            "Not applicable", 'No', inplace=True)
        data_frame['Nicotine replacement product used to help stop smoking: Nicotine patch (capi+casi)'].replace(
            "Not applicable", 'No', inplace=True)
        data_frame['Nicotine replacement product used to help stop smoking: Nicotine inhaler (capi+casi)'].replace(
            "Not applicable", 'No', inplace=True)
        data_frame['Nicotine replacement product used to help stop smoking: Nicotine mouth spray (capi+casi)'].replace(
            "Not applicable", 'No', inplace=True)
        data_frame['Nicotine replacement product used to help stop smoking: Nicotine nasal spray (capi+casi)'].replace(
            "Not applicable", 'No', inplace=True)
        data_frame[
            'Nicotine replacement product used to help stop smoking: Another nicotine product (capi+casi)'].replace(
            "Not applicable", 'No', inplace=True)
        data_frame[
            'Nicotine replacement product used to help stop smoking: Electronic cigarette or vaping device (capi+casi)'].replace(
            "Not applicable", 'No', inplace=True)
        data_frame['Nicotine replacement product used to help stop smoking: None of these (capi+casi)'].replace(
            "Not applicable", 'No', inplace=True)

        data_frame['Currently trying to cut down on smoking but not trying to stop (Capi)'].replace("Not applicable",
                                                                                                    'Not smoker',
                                                                                                    inplace=True)
        data_frame[
            'Nicotine replacement product currently used to cut down the amount smoked: Nicotine chewing gum (capi+casi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)
        data_frame[
            'Nicotine replacement product currently used to cut down the amount smoked: Nicotine lozenges (capi+casi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)
        data_frame[
            'Nicotine replacement product currently used to cut down the amount smoked: Nicotine patch (capi+casi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)

        data_frame[
            'Nicotine replacement product currently used to cut down the amount smoked: Nicotine inhaler (capi+casi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)
        data_frame[
            'Nicotine replacement product currently used to cut down the amount smoked: Nicotine mouth spray (capi+casi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)
        data_frame[
            'Nicotine replacement product currently used to cut down the amount smoked: Nicotine nasal spray (capi+casi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)
        data_frame[
            'Nicotine replacement product currently used to cut down the amount smoked: Another nicotine product (capi+casi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)

        data_frame[
            'Nicotine replacement product currently used to cut down the amount smoked: Electronic cigarette or vaping device (capi+casi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)
        data_frame[
            'Nicotine replacement product currently used to cut down the amount smoked: None of these (capi+casi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)
        data_frame[
            'Nicotine replacement product used in situations where not allowed to smoke: Nicotine chewing gum (capi+casi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)
        data_frame[
            'Nicotine replacement product used in situations where not allowed to smoke: Nicotine lozenges (capi+casi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)

        data_frame[
            'Nicotine replacement product used in situations where not allowed to smoke: Nicotine patch (capi+casi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)
        data_frame[
            'Nicotine replacement product used in situations where not allowed to smoke: Nicotine inhaler (capi+casi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)
        data_frame[
            'Nicotine replacement product used in situations where not allowed to smoke: Nicotine mouth spray (capi+casi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)
        data_frame[
            'Nicotine replacement product used in situations where not allowed to smoke: Nicotine nasal spray (capi+casi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)

        data_frame[
            'Nicotine replacement product used in situations where not allowed to smoke: Another nicotine product (capi+casi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)
        data_frame[
            'Nicotine replacement product used in situations where not allowed to smoke: Electronic cigarette or vaping device (capi+casi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)
        data_frame[
            'Nicotine replacement product used in situations where not allowed to smoke: None of these (capi+casi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)
        data_frame[
            'Nicotine replacement products used to help stop smoking during serious quit attempt: Nicotine chewing gum (capi+casi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)

        data_frame[
            'Nicotine replacement products used to help stop smoking during serious quit attempt: Nicotine lozenges (capi+casi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)
        data_frame[
            'Nicotine replacement products used to help stop smoking during serious quit attempt: Nicotine patch (capi+casi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)
        data_frame[
            'Nicotine replacement products used to help stop smoking during serious quit attempt: Nicotine inhaler (capi+casi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)
        data_frame[
            'Nicotine replacement products used to help stop smoking during serious quit attempt: Nicotine mouth spray (capi+casi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)

        data_frame[
            'Nicotine replacement products used to help stop smoking during serious quit attempt: Nicotine nasal spray (capi+casi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)
        data_frame[
            'Nicotine replacement products used to help stop smoking during serious quit attempt: Another nicotine product (capi+casi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)
        data_frame[
            'Nicotine replacement products used to help stop smoking during serious quit attempt: Electronic cigarette or vaping device (capi+casi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)
        data_frame[
            'Nicotine replacement products used to help stop smoking during serious quit attempt: None of these (capi+casi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)

        data_frame[
            'In the last 12 months medical person advised to stop smoking completely (Capi)'] \
            .replace("Not applicable", 'Not smoker', inplace=True)
        data_frame[
            'Currently smokes cigars (capi+casi)'] \
            .replace("Not applicable", 'Not cigars smoker', inplace=True)
        data_frame[
            'How regularly smokes cigars (Capi)'] \
            .replace("Not applicable", 'Not cigar smoker', inplace=True)
        data_frame[
            'Currently smokes a pipe (capi+casi)'] \
            .replace("Not applicable", 'Not pipe smoker', inplace=True)

        # on car
        data_frame['Number of cars normally available'].replace("Not applicable", 'none', inplace=True)

        # BMI
        data_frame['(D) Valid BMI measurements using estimated weight if >130kg'] \
            = np.where(data_frame['(D) Self-reported BMI'] != 'Not applicable', data_frame['(D) Self-reported BMI'],
                       data_frame['(D) Valid BMI measurements using estimated weight if >130kg'])

        DataController.convert_by_median(data_frame, '(D) Valid BMI measurements using estimated weight if >130kg')
        data_frame.drop(columns=['(D) Self-reported BMI'], inplace=True)

        # on HRP
        data_frame['HRP: How long have you been looking/were you looking for paid employment'].replace("Not applicable",
                                                                                                       'Not looking',
                                                                                                       inplace=True)
        data_frame['HRP: Whether working full-time or part-time'].replace("Not applicable", 'Not employed',
                                                                          inplace=True)
        data_frame['HRP: Whether an employee or self-employed'].replace("Not applicable", 'Not employed', inplace=True)
        data_frame['HRP: Whether a manager or foreman'].replace("Not applicable", 'Not employed', inplace=True)
        data_frame['(D) HRP: if self-employed do/did you have any employees? (Top coded)'].replace("Not applicable",
                                                                                                   'Not self-employed',
                                                                                                   inplace=True)

        data_frame['Income: Earnings from employment or self-employment'].replace("Not applicable", 'Not mentioned',
                                                                                  inplace=True)
        data_frame['Income: State retirement pension'].replace("Not applicable", 'Not mentioned', inplace=True)
        data_frame['Income: Pension from former employer'].replace("Not applicable", 'Not mentioned', inplace=True)
        data_frame['Income: Personal pensions'].replace("Not applicable", 'Not mentioned', inplace=True)
        data_frame['Income: Job-Seekers Allowance'].replace("Not applicable", 'Not mentioned', inplace=True)
        data_frame['Income: Employment and Support Allowance'].replace("Not applicable", 'Not mentioned', inplace=True)
        data_frame['Income: Income Support'].replace("Not applicable", 'Not mentioned', inplace=True)
        data_frame['Income: Pension Credit'].replace("Not applicable", 'Not mentioned', inplace=True)
        data_frame['Income: Working Tax Credit'].replace("Not applicable", 'Not mentioned', inplace=True)
        data_frame['Income: Child Tax Credit'].replace("Not applicable", 'Not mentioned', inplace=True)
        data_frame['Income: Child Benefit'].replace("Not applicable", 'Not mentioned', inplace=True)
        data_frame['Income: Housing Benefit'].replace("Not applicable", 'Not mentioned', inplace=True)
        data_frame['Income: Council Tax Benefit'].replace("Not applicable", 'Not mentioned', inplace=True)
        data_frame['Income: Universal Credit'].replace("Not applicable", 'Not mentioned', inplace=True)
        data_frame['Income: Other state benefits'].replace("Not applicable", 'Not mentioned', inplace=True)
        data_frame['Income: Interest from savings and investments (e.g. stocks & shares)'].replace("Not applicable",
                                                                                                   'Not mentioned',
                                                                                                   inplace=True)
        data_frame[
            'Income: Other kinds of regular allowance from outside your household (e.g. maintenance, student grants, rent)'].replace(
            "Not applicable", 'Not mentioned', inplace=True)

        # on landlord
        data_frame['Who is your landlord'].replace("Not applicable", 'no Landlord', inplace=True)

        data_frame['Is the accommodation furnished'].replace("Not applicable", 'Not rented', inplace=True)

        data_frame['Consent to store blood for future analysis'].replace("Not applicable", 'No bloods to store',
                                                                         inplace=True)
        data_frame['Blood sample not to GP: Hardly/never sees GP'].replace("Not applicable", 'Not mentioned',
                                                                           inplace=True)
        data_frame['Blood sample not to GP: GP recently took blood sample'].replace("Not applicable", 'Not mentioned',
                                                                                    inplace=True)
        data_frame['Blood sample not to GP: Does not want to bother GP'].replace("Not applicable", 'Not mentioned',
                                                                                 inplace=True)
        data_frame['Blood sample not to GP: Other reason'].replace("Not applicable", 'Not mentioned', inplace=True)
        data_frame['Permission to send results of blood sample to GP'].replace("Not applicable", 'No bloods to send',
                                                                               inplace=True)
        data_frame[
            '(D) Diabetes from blood sample (48+mmol/mol) or doctor diagnosis (excluding pregnancy-only diabetes)'].replace(
            "Not applicable", 'No usable sample', inplace=True)
        data_frame[
            '(D) Total diabetes from blood sample or doctor diagnosis (excluding pregnancy-only diabetes) revised'].replace(
            "Not applicable", 'No usable sample', inplace=True)
        data_frame[
            '(D) Total diabetes from blood sample or doctor diagnosis (excluding pregnancy-only diabetes)'].replace(
            "Not applicable", 'No usable sample', inplace=True)
        data_frame[
            '(D) Total diabetes from blood sample or doctor diagnosis'].replace(
            "Not applicable", 'No usable sample', inplace=True)
        data_frame[
            '(D) Diabetes from blood sample or doctor diagnosis (excluding pregnancy-only diabetes) revised (comparable with pre-September 2013)'].replace(
            "Not applicable", 'No usable sample', inplace=True)
        data_frame[
            '(D) Diabetes from blood sample or doctor diagnosis (excluding pregnancy-only diabetes) revised'].replace(
            "Not applicable", 'No usable sample', inplace=True)
        data_frame[
            '(D) Diabetes from blood sample or doctor diagnosis (excluding pregnancy-only diabetes)'].replace(
            "Not applicable", 'No usable sample', inplace=True)

        data_frame['Outcome of blood sample'].replace("Not applicable", 'Refused', inplace=True)

        list_of_blood_q = ['No blood obtained: Other reason', 'Blood sample outcome',
                           'Blood sample prob: Some blood obtained but respondent felt faint/fainted',
                           'Blood sample prob: Unable to use tourniquet',
                           'No blood obtained: No suitable/palpable vein/collapsed veins',
                           'No blood obtained: Respondent was too anxious/nervous',
                           'No blood obtained: Respondent felt faint/fainted', 'No blood obtained: Other reason',
                           'Whether wants results of blood sample sent',
                           'Refused blood sample: Previous difficulties with venepuncture',
                           'Refused blood sample: Dislike/fear of needles',
                           'Refused blood sample: Respondent recently had blood test/health check',
                           'Refused blood sample: Current illness',
                           'Refused blood sample: Worried about HIV or Aids', 'Refused blood sample: Other reason',
                           'Total cholesterol serum quality', 'HDL cholesterol serum quality',
                           'Glycated haemoglobin serum quality',
                           'Glycated haemoglobin serum quality (mmol/mol)', 'Sample not obtained: Other',
                           'Consent to blood sample']

        DataController.replace_based_on_condition(data_frame, list_of_blood_q, 'Outcome of blood sample', 'Refused',
                                                  'blood sample refused')
        DataController.replace_based_on_condition(data_frame, list_of_blood_q, 'Outcome of blood sample',
                                                  'Taken (at least a tube)', 'blood sample taken')
        DataController.replace_based_on_condition(data_frame, list_of_blood_q, 'Outcome of blood sample',
                                                  'No blood sample obtained', 'No blood sample obtained')
        DataController.replace_based_on_condition(data_frame, list_of_blood_q, 'Outcome of blood sample',
                                                  'Not applicable (pregnant/Warfarin/epilepsy etc.)',
                                                  'Not applicable (pregnant/Warfarin/epilepsy etc.)')
        DataController.replace_based_on_condition(data_frame, list_of_blood_q, 'Outcome of blood sample',
                                                  'Not attempted', 'Not attempted')

        data_frame[
            'Blood sample prob: Some blood obtained but respondent felt faint/fainted'].replace("Not applicable",
                                                                                                'Not mentioned',
                                                                                                inplace=True)
        data_frame[
            'Blood sample prob: Unable to use tourniquet'].replace("Not applicable", 'Not mentioned', inplace=True)

        data_frame[
            'Refused blood sample: Previous difficulties with venepuncture'].replace("Not applicable", 'Not mentioned',
                                                                                     inplace=True)
        data_frame[
            'Refused blood sample: Dislike/fear of needles'].replace("Not applicable", 'Not mentioned', inplace=True)
        data_frame[
            'Refused blood sample: Respondent recently had blood test/health check'].replace("Not applicable",
                                                                                             'Not mentioned',
                                                                                             inplace=True)
        data_frame[
            'Refused blood sample: Current illness'].replace("Not applicable", 'Not mentioned', inplace=True)
        data_frame[
            'Refused blood sample: Worried about HIV or Aids'].replace("Not applicable", 'Not mentioned', inplace=True)
        data_frame[
            'Refused blood sample: Other reason'].replace("Not applicable", 'Not mentioned', inplace=True)

        data_frame[
            'Whether wants results of blood sample sent'].replace("Not applicable", 'No blood sample obtained',
                                                                  inplace=True)

        DataController.convert_by_median(data_frame, 'Glycated haemoglobin result (mmol/mol)')
        DataController.convert_by_median(data_frame, 'Glycated haemoglobin result (%)')
        DataController.convert_by_median(data_frame,
                                         '(D) Valid Glycated haemoglobin result in mmol per ml (IFFC) (comparable with pre-September 2013)')
        DataController.convert_by_median(data_frame, '(D) Valid Glycated haemoglobin result in mmol/ml (IFFC)')
        DataController.convert_by_median(data_frame,
                                         '(D) Valid Glycated haemoglobin result (%) (comparable with pre-September 2013)')
        DataController.convert_by_median(data_frame, '(D) Valid Glycated haemoglobin result (%)')
        DataController.convert_by_median(data_frame,
                                         '(D) Valid HDL Cholesterol Result mmol/L (incl those on LLD) (comparable with pre-2010 results)')
        DataController.convert_by_median(data_frame, '(D) Valid HDL Cholesterol result (incl those on LLD)')
        DataController.convert_by_median(data_frame,
                                         '(D) Valid Total Cholesterol Result mmol/L (incl those on LLD) (comparable with pre-2010 results)')
        DataController.convert_by_median(data_frame, '(D) Valid Total Cholesterol result (incl those on LLD)')
        DataController.convert_by_median(data_frame, 'Total cholesterol result (mmol/L)')
        DataController.convert_by_median(data_frame, 'HDL Cholesterol result')

        data_frame['Blood sample prob: Some blood obtained but respondent felt faint/fainted']\
            .replace("Not applicable", 'Not mentioned', inplace=True)

        data_frame['Blood sample prob: Unable to use tourniquet'].replace("Not applicable", 'Not mentioned',
                                                                          inplace=True)
        data_frame['No blood obtained: No suitable/palpable vein/collapsed veins'].replace("Not applicable",
                                                                                           'Not mentioned',
                                                                                           inplace=True)
        data_frame['No blood obtained: Respondent was too anxious/nervous'].replace("Not applicable", 'Not mentioned',
                                                                                    inplace=True)
        data_frame['No blood obtained: Respondent felt faint/fainted'].replace("Not applicable", 'Not mentioned',
                                                                               inplace=True)
        data_frame['No blood obtained: Other reason'].replace("Not applicable", 'Not mentioned', inplace=True)

        data_frame['Refused blood sample: Dislike/fear of needles'].replace("Not applicable", 'Not mentioned',
                                                                            inplace=True)
        data_frame['Refused blood sample: Respondent recently had blood test/health check'].replace("Not applicable",
                                                                                                    'Not mentioned',
                                                                                                    inplace=True)
        data_frame['Refused blood sample: Current illness'].replace("Not applicable", 'Not mentioned', inplace=True)
        data_frame['Refused blood sample: Worried about HIV or Aids'].replace("Not applicable", 'Not mentioned',
                                                                              inplace=True)
        data_frame['Refused blood sample: Other reason'].replace("Not applicable", 'Not mentioned', inplace=True)

        list_of_Saliva_q = ['Sample not obtained: Other', 'Sample not obtained: Not able to produce any saliva',
                            'Method used to obtain saliva sample', 'Whether saliva sample obtained']

        DataController.replace_based_on_condition(data_frame, list_of_Saliva_q, 'Saliva sample outcome', 'Refused',
                                                  'Saliva sample refused')
        DataController.replace_based_on_condition(data_frame, list_of_Saliva_q, 'Saliva sample outcome', 'Obtained',
                                                  'Saliva sample Obtained')

        data_frame['Whether saliva sample obtained'].replace("Not applicable", 'Saliva sample not obtained', inplace=True)

        DataController.convert_by_median(data_frame, '(D) WEMWBS Score')

        data_frame['Reason no BP measurements taken'].replace('Not applicable', 'BP measurements taken', inplace=True)
        data_frame['BP not to GP: Other reason'].replace('Not applicable', 'Not mentioned', inplace=True)
        data_frame['BP not to GP: Does not want to bother GP'].replace('Not applicable', 'Not mentioned', inplace=True)
        data_frame["BP not to GP: GP knows respondent's BP"].replace('Not applicable', 'Not mentioned', inplace=True)
        data_frame['BP not to GP: Hardly/never sees GP'].replace('Not applicable', 'Not mentioned', inplace=True)
        data_frame['Consent to send BP readings to GP'].replace('Not applicable', 'No reading to give', inplace=True)

        list_of_bp_q = ['(D) Hypertensive categories: all taking BP drugs (Omron readings) revised',
                        '(D) Whether hypertensive: all taking BP drugs (Omron readings) revised',
                        '(D) Hypertensive categories:140/90: all prescribed drugs for BP (Omron readings) revised',
                        '(D) Whether hypertensive:140/90: all prescribed drugs for BP (Omron readings) revised',
                        '(D) Hypertensive categories: all prescribed drugs for BP (Omron readings) revised',
                        '(D) Whether hypertensive: all prescribed drugs for BP (Omron readings) revised',
                        '(D) Valid blood pressure 3 groups',
                        '(D) Hypertensive untreated (160/100): all prescribed drugs for BP (Omron readings) revised',
                        '(D) SBP in 5 groups',
                        '(D) Hypertensive untreated: all prescribed drugs for BP (Omron readings) revised',
                        'BP not obtained: Problems with PC', 'BP not obtained: Respondent upset/anxious/nervous',
                        'BP not obtained: Error reading',
                        'BP not obtained: Respondent too shy',
                        'BP not obtained: Child would not sit still',
                        'BP not obtained: Problems with cuff fitting/painful',
                        'BP not obtained: Problems with equipment',
                        'BP not obtained: Other reason',
                        'BP problems: No problems taking blood pressure',
                        'BP problems: Reading on left arm as right arm not suitable',
                        'BP problems: Respondent was anxious/upset/nervous',
                        'BP problems: Problem with cuff fitting/painful',
                        'BP problems: Omron problem (not error reading)',
                        'BP problems: Omron error reading',
                        'BP problems: Other problem']
        DataController.replace_based_on_condition(data_frame, list_of_bp_q, '(D) Whether BP readings are valid',
                                                  'Ate, drank, smoked, exercised in previous half hour',
                                                  'Ate, drank, smoked, exercised in previous half hour')
        DataController.replace_based_on_condition(data_frame, list_of_bp_q, '(D) Whether BP readings are valid',
                                                  'Refused, attempted but not obtained, not attempted',
                                                  'Refused, attempted but not obtained, not attempted')
        DataController.replace_based_on_condition(data_frame, list_of_bp_q, '(D) Whether BP readings are valid',
                                                  'Three valid readings not obtained',
                                                  'Three valid readings not obtained')

        DataController.convert_by_median(data_frame, '1st pulse reading (bpm)')
        DataController.convert_by_median(data_frame, '1st Diastolic reading (mmHg)')
        DataController.convert_by_median(data_frame, '1st Systolic reading (mmHg)')
        DataController.convert_by_median(data_frame, '1st MAP reading (mmHg)')
        DataController.convert_by_median(data_frame, '2nd Systolic reading (mmHg)')
        DataController.convert_by_median(data_frame, '2nd Diastolic reading (mmHg)')
        DataController.convert_by_median(data_frame, '2nd pulse reading (bpm)')
        DataController.convert_by_median(data_frame, '2nd MAP reading (mmHg)')
        DataController.convert_by_median(data_frame, '3rd Systolic reading (mmHg)')
        DataController.convert_by_median(data_frame, '3rd Diastolic reading (mmHg)')
        DataController.convert_by_median(data_frame, '3rd pulse reading (bpm)')
        DataController.convert_by_median(data_frame, '3rd MAP reading (mmHg)')

        list_of_bp_q = ['BP not obtained: Problems with PC',
                        'BP not obtained: Respondent upset/anxious/nervous',
                        'BP not obtained: Error reading',
                        'BP not obtained: Respondent too shy',
                        'BP not obtained: Child would not sit still',
                        'BP not obtained: Problems with cuff fitting/painful',
                        'BP not obtained: Problems with equipment',
                        'BP not obtained: Other reason']

        DataController.replace_based_on_condition(data_frame, list_of_bp_q, 'Reason no BP measurements taken',
                                                  'BP measurements taken', 'BP measurements taken')

        # remove help columns
        data_frame['(D) Needed help with any personal activities (ADLs excl bath or shower, toilet, indoors & stairs)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame['(D) Needed help with any personal activities (ADLs excl bath or shower)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) Needed help with any indoor activities (ADLs: Getting around indoors, getting up and down stairs)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame['(D) Needed help with any personal activities (ADLs)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) Needed help with any instrumental activities (IADLs: getting out of house, food shopping, routine housework, doing paperwork/bills)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame['(D) Received help for any personal activities (ADLs)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame['(D) Received help for any personal activities (ADLs excl bath or shower)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame['(D) Received help for any personal activities (ADLs excl bath or shower, toilet, indoors & stairs)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) Received help with any indoor activities (ADLs: Getting around indoors, getting up and down stairs)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) Received help with any instrumental activities (IADLs: getting out of house, food shopping, routine housework, doing paperwork/bills)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) Received help with ADLs/IADLs in the last month'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Received help: Stairs (binary) (TASK I)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Received help: Indoors (binary) (TASK H)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Received help: Bed (binary) (TASK A)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Received help: Shower (binary) (TASK C)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Received help: Dress (binary) (TASK D)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) Received help: Wash (binary) (TASK B)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Received help: Toilet (binary) (TASK E)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Received help: Medicine (binary) (TASK G)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Received help: Eat (binary) (TASK F)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) Received help: House (binary) (TASK J)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Received help: Shop (binary) (TASK K)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Received help: Housework (binary) (TASK L)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Received help: Paperwork (binary) (TASK M)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) Payment for care'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Who provided help with ADLs or IADLs in the last month'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            'Person 1 provide help/ support to: lives in household'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            '(D) Person 1 provide help/ support to: age (recoded)'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Person 1 provide help/ support to: sex'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 2 provide help/ support to: lives in household'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            '(D) Person 2 provide help/ support to: age (recoded)'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 2 provide help/ support to: sex'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Person 3 provide help/ support to: lives in household'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            '(D) Person 3 provide help/ support to: age (recoded)'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 3 provide help/ support to: sex'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 provide help/ support to: number of hours provided help in the last week'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Person 1 provide help/ support to: number of hours provided help in a usual week'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            'Person 2 provide help/ support to: number of hours provided help in the last week'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            'Person 2 provide help/ support to: number of hours provided help in a usual week'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 3 provide help/ support to: number of hours provided help in the last week'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Person 3 provide help/ support to: number of hours provided help in a usual week'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Approximate time spent providing support or help to those aged 65+'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 activities provided help or support with: Getting the person in and out of bed'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 activities provided help or support with: Washing their face and hands'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Person 1 activities provided help or support with: Having a bath or a shower'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 activities provided help or support with: Dressing or undressing'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 activities provided help or support with: Using the toilet'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 activities provided help or support with: Eating (including cutting up food)'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Person 1 activities provided help or support with: Taking the right amount of medicine at the right times'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 activities provided help or support with: Getting around indoors'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 activities provided help or support with: Getting up and down the stairs'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 activities provided help or support with: Getting out of the house'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Person 1 activities provided help or support with: Shopping for food'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 activities provided help or support with: Doing routine housework or laundry'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 activities provided help or support with: Doing paperwork or paying bills'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Payment for care person 1: Yes, this person pays me from their own income, pensions or savings'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Payment for care person 1: Yes, this person pays me from a personal budget or direct payment'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Payment for care person 1: Yes, I receive a carer’s allowance'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Payment for care person 1: Yes, I receive money in another way'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Payment for care person 1: No, I receive no money for helping this person'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            '(D) Person 1: Amount received per week (grouped)'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1: Frequency of payment (per week or month)'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 2 activities provided help or support with: Getting the person in and out of bed'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 2 activities provided help or support with: Washing their face and hands'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Person 2 activities provided help or support with: Having a bath or a shower'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 2 activities provided help or support with: Dressing or undressing'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 2 activities provided help or support with: Using the toilet'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 2 activities provided help or support with: Eating (including cutting up food)'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Person 2 activities provided help or support with: Taking the right amount of medicine at the right times'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 2 activities provided help or support with: Getting around indoors'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 2 activities provided help or support with: Getting up and down the stairs'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 2 activities provided help or support with: Getting out of the house'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Person 2 activities provided help or support with: Shopping for food'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 2 activities provided help or support with: Doing routine housework or laundry'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 2 activities provided help or support with: Doing paperwork or paying bills'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Payment for care person 2: Yes, this person pays me from their own income, pensions or savings'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Payment for care person 2: Yes, this person pays me from a personal budget or direct payment'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Payment for care person 2: Yes, I receive a carer’s allowance'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Payment for care person 2: Yes, I receive money in another way'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Payment for care person 2: No, I receive no money for helping this person'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            '(D) Person 2: Amount received per week (grouped)'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 2: Frequency of payment (per week or month)'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 3 activities provided help or support with: Getting the person in and out of bed'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 3 activities provided help or support with: Washing their face and hands'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Person 3 activities provided help or support with: Having a bath or a shower'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 3 activities provided help or support with: Dressing or undressing'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 3 activities provided help or support with: Using the toilet'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 3 activities provided help or support with: Eating (including cutting up food)'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Person 3 activities provided help or support with: Taking the right amount of medicine at the right times'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 3 activities provided help or support with: Getting around indoors'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 3 activities provided help or support with: Getting up and down the stairs'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 3 activities provided help or support with: Getting out of the house'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Person 3 activities provided help or support with: Shopping for food'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 3 activities provided help or support with: Doing routine housework or laundry'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 3 activities provided help or support with: Doing paperwork or paying bills'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Payment for care person 3: Yes, this person pays me from their own income, pensions or savings'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 3: About how long been looking after or helping person cared for'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Payment for care person 3: Yes, this person pays me from a personal budget or direct payment'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Payment for care person 3: Yes, I receive a carer’s allowance'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Payment for care person 3: Yes, I receive money in another way'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Payment for care person 3: No, I receive no money for helping this person'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            '(D) Person 3: Amount received per week (grouped)'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 3: Frequency of payment (per week or month)'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Support received for caring for person 1: Help from GP or nurse'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Support received for caring for person 1: Access to respite care'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Support received for caring for person 1: Help from professional care staff'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Support received for caring for person 1: Help from carers’ organisation or charity'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Support received for caring for person 1: Help from other family members'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Support received for caring for person 1: Advice from local authority/ social services'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Support received for caring for person 1: Help from friends/neighbours'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Support received for caring for person 1: No support/ help received from any of these'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Support received for caring for person 2: Help from GP or nurse'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Support received for caring for person 2: Access to respite care'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Support received for caring for person 2: Help from professional care staff'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Support received for caring for person 2: Help from carers’ organisation or charity'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Support received for caring for person 2: Help from other family members'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Support received for caring for person 2: Advice from local authority/ social services'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Support received for caring for person 2: Help from friends/neighbours'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Support received for caring for person 2: No support/ help received from any of these'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Support received for caring for person 3: Help from GP or nurse'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Support received for caring for person 3: Access to respite care'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Support received for caring for person 3: Help from professional care staff'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Support received for caring for person 3: Help from carers’ organisation or charity'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Support received for caring for person 3: Help from other family members'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Support received for caring for person 3: Advice from local authority/ social services'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Support received for caring for person 3: Help from friends/neighbours'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Support received for caring for person 3: No support/ help received from any of these'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Health affected in the last three months by help/ support provided: Feeling tired'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            '(D) Grouped hours provided (for care recipient for whom most hours provided)'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            '(D) Support received for caring: Help from GP or nurse'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            '(D) Support received for caring: Access to respite care'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            '(D) Support received for caring: Help from professional care staff'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            '(D) Support received for caring: Help from carers’ organisation or charity'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            '(D) Support received for caring: Help from other family members'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            '(D) Support received for caring: Advice from local authority/ social services'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            '(D) Support received for caring: Help from friends/neighbours'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            '(D) Support received for caring: None of these'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            '(D) Who is cared for: Husband/Wife/Partner'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            '(D) Who is cared for: Mother (including mother-in-law)'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            '(D) Who is cared for: Father (including father-in-law)'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            '(D) Who is cared for: Son or daughter (including stepchildren, adopted children or children-in-law)'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            '(D) Who is cared for: Grandparent'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            '(D) Who is cared for: Grandchild (including great grandchildren)'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            '(D) Who is cared for: Brother or sister'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            '(D) Who is cared for: Other family member'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            '(D) Who is cared for: Friend'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            '(D) Who is cared for: Neighbour'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            '(D) Who is cared for: Someone else'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            '(D) Cares for someone in the same household'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            '(D) Cares for someone in a different household'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            '(D) Grouped hours provided 20+ with no help and less than 1 hour combined (for care recipient for whom most hours provided)'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            '(D) Grouped hours provided 10+ with no help and less than 1 hour combined (for care recipient for whom most hours provided)'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            '(D) Grouped hours provided 10+, 2 groups (for care recipient for whom most hours provided)'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Thinking about the other people you have caring responsibilities for, which of the following best describes your current situation'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1: Last month helped with: Personal care'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1: Last month helped with: Physical help'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1: Last month helped with: Dealing with care services and benefits'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Person 1: Last month helped with: Other paperwork or financial matters'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1: Last month helped with: Other practical help (e.g. shopping for food)'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1: Last month helped with: Keeping him/her company'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1: Last month helped with: Taking him/her out'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Person 1: Last month helped with: Giving medicines'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1: Last month helped with: Keeping an eye on him/her to see he/she is alright'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1: Last month helped with: None of these'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1: How often helped with personal care'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Person 1: How often helped with physical help'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1: How often helped with dealing with care services and benefits'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1: How often helped with other paperwork or financial matters'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1: How often helped with other practical help'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Person 1: How often helped with keeping him/her company'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1: How often helped with taking him/her out'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1: How often helped with giving medicines'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1: How often helped with keeping an eye on him/her to see if he/she is alright'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Person 1: About how long been looking after or helping person 1'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 cared for receives regular visits at least once a month from: Doctor'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 cared for receives regular visits at least once a month from: Community/ district nurse/ Community Matron'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 cared for receives regular visits at least once a month from: Health visitor'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Person 1 cared for receives regular visits at least once a month from: Social worker/ care manager'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 cared for receives regular visits at least once a month from: Home help/ care worker'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 cared for receives regular visits at least once a month from: Meals on wheels'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 cared for receives regular visits at least once a month from: Voluntary worker'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Person 1 cared for receives regular visits at least once a month from: Occupational therapist'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 cared for receives regular visits at least once a month from: Specialist/nursing care/ palliative care'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 cared for receives regular visits at least once a month from: Community mental health services'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 cared for receives regular visits at least once a month from: Gardener/ caretaker/ warden'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Person 1 cared for receives regular visits at least once a month from: Other professional visitor'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 cared for receives regular visits at least once a month from: None'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 cared for has regular contact from: Doctor'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 cared for has regular contact from: Community/ district nurse/ community matron'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Person 1 cared for has regular contact from: Health visitor'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 cared for has regular contact from: Social worker/ care manager'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 cared for has regular contact from: Home help/ care worker'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 cared for has regular contact from: Meals on wheels'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Person 1 cared for has regular contact from: Voluntary worker'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 cared for has regular contact from: Occupational therapist'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 cared for has regular contact from: Educational professional'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 cared for has regular contact from: Specialist/nursing care/ palliative care'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Person 1 cared for has regular contact from: Community mental health services'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 cared for has regular contact from: Gardener/ caretaker/ warden'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 cared for has regular contact from: Other professional visitor'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 cared for has regular contact from: None'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Reasons for person 1 not receiving regular visits at least once a month: Not available/not offered'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Reasons for person 1 not receiving regular visits at least once a month: Not needed'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Reasons for person 1 not receiving regular visits at least once a month: Tried, but not helpful'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Reasons for person 1 not receiving regular visits at least once a month: Not wanted by respondent'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Reasons for person 1 not receiving regular visits at least once a month: Not at a convenient time'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Reasons for person 1 not receiving regular visits at least once a month: Too expensive'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Reasons for person 1 not receiving regular visits at least once a month: Not eligible'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reasons for person 1 not receiving regular visits at least once a month: Don't know who to ask"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Reasons for person 1 not receiving regular visits at least once a month: Other'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Person 1 cared for regularly makes use of a community or voluntary transport scheme'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Reason started looking after or giving special help to person 1: No one else available'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reason started looking after or giving special help to person 1: Was willing / wanted to help out"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Reason started looking after or giving special help to person 1: Had the time because was not working'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Reason started looking after or giving special help to person 1: Had the time because was working part time'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Reason started looking after or giving special help to person 1: Have particular skills / ability to care'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reason started looking after or giving special help to person 1: Social Services (Local Authority) suggested respondent should provide care"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Reason started looking after or giving special help to person 1: It was expected of respondent (it's what families do)"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reason started looking after or giving special help to person 1: He/she wouldn't want anyone else caring for them"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reason started looking after or giving special help to person 1: Cared for person requested respondent's help/care"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reason started looking after or giving special help to person 1: Took over caring responsibilities from someone else"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Reason started looking after or giving special help to person 1: They needed help (person cared for)"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reason started looking after or giving special help to person 1: Other (SPECIFY)"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Assistance/ caring for person 1 affected ability to spend time doing leisure or social activities: Unable to socialise"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Assistance/ caring for person 1 affected ability to spend time doing leisure or social activities: Reduced time with spouse or partner"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Assistance/ caring for person 1 affected ability to spend time doing leisure or social activities: Reduced time with other family members"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Assistance/ caring for person 1 affected ability to spend time doing leisure or social activities: Reduced time with friends"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Assistance/ caring for person 1 affected ability to spend time doing leisure or social activities: Difficulties making new friends"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Assistance/ caring for person 1 affected ability to spend time doing leisure or social activities: Reduced time spent doing sport or physical activity"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Assistance/ caring for person 1 affected ability to spend time doing leisure or social activities: Reduced time spent doing pastime or hobby"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Assistance/ caring for person 1 affected ability to spend time doing leisure or social activities: Other"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Assistance/ caring for person 1 affected ability to spend time doing leisure or social activities: None of these"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2: Last month helped with: Personal care"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 2: Last month helped with: Dealing with care services and benefits"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2: Last month helped with: Other paperwork or financial matters"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2: Last month helped with: Other practical help (e.g. shopping for food)"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2: Last month helped with: Keeping him/her company"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 2: Last month helped with: Taking him/her out"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2: Last month helped with: Giving medicines"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2: Last month helped with: Keeping an eye on him/her to see he/she is alright"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2: Last month helped with: None of these"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 2: How often helped with personal care"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2: How often helped with physical help"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2: How often helped with dealing with care services and benefits"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2: How often helped with other paperwork or financial matters"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 2: How often helped with other practical help"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2: How often helped with keeping him/her company"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2: How often helped with taking him/her out"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2: How often helped with giving medicines"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 2: How often helped with keeping an eye on him/her to see if he/she is alright"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2: About how long been looking after or helping person 2"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2 cared for receives regular visits at least once a month from: Doctor"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2 cared for receives regular visits at least once a month from: Community/ district nurse/ Community Matron"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 2 cared for receives regular visits at least once a month from: Health visitor"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2 cared for receives regular visits at least once a month from: Social worker/ care manager"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2 cared for receives regular visits at least once a month from: Home help/ care worker"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2 cared for receives regular visits at least once a month from: Meals on wheels"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 2 cared for receives regular visits at least once a month from: Voluntary worker"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2 cared for receives regular visits at least once a month from: Occupational therapist"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2 cared for receives regular visits at least once a month from: Educational professional"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2 cared for receives regular visits at least once a month from: Specialist/nursing care/ palliative care"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 2 cared for receives regular visits at least once a month from: Community mental health services"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2 cared for receives regular visits at least once a month from: Gardener/ caretaker/ warden"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2 cared for receives regular visits at least once a month from: Other professional visitor"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2 cared for receives regular visits at least once a month from: None"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 2 cared for has regular contact from: Doctor"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2 cared for has regular contact from: Community/ district nurse/ community matron"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2 cared for has regular contact from: Health visitor"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2 cared for has regular contact from: Social worker/ care manager"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 2 cared for has regular contact from: Home help/ care worker"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2 cared for has regular contact from: Meals on wheels"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2 cared for has regular contact from: Voluntary worker"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2 cared for has regular contact from: Occupational therapist"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 2 cared for has regular contact from: Educational professional"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2 cared for has regular contact from: Specialist/nursing care/ palliative care"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2 cared for has regular contact from: Community mental health services"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2 cared for has regular contact from: Gardener/ caretaker/ warden"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 2 cared for has regular contact from: Other professional visitor"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 2 cared for has regular contact from: None"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reasons for person 2 not receiving regular visits at least once a month: Not available/not offered"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reasons for person 2 not receiving regular visits at least once a month: Not needed"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Reasons for person 2 not receiving regular visits at least once a month: Tried, but not helpful"].replace(
            'Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Reasons for person 2 not receiving regular visits at least once a month: Not wanted by respondent"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reasons for person 2 not receiving regular visits at least once a month: Not wanted by the person respondent cares for"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reasons for person 2 not receiving regular visits at least once a month: Not at a convenient time"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Reasons for person 2 not receiving regular visits at least once a month: Too expensive"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reasons for person 2 not receiving regular visits at least once a month: Not eligible"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reasons for person 2 not receiving regular visits at least once a month: Don't know who to ask"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reasons for person 2 not receiving regular visits at least once a month: Other"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 2 cared for regularly makes use of a community or voluntary transport scheme"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reason started looking after or giving special help to person 2: No one else available"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reason started looking after or giving special help to person 2: Was willing / wanted to help out"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reason started looking after or giving special help to person 2: Had the time because was not working"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Reason started looking after or giving special help to person 2: Had the time because was working part time"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reason started looking after or giving special help to person 2: Have particular skills / ability to care"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reason started looking after or giving special help to person 2: Social Services (Local Authority) suggested respondent should provide care"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reason started looking after or giving special help to person 2: It was expected of respondent (it's what families do)"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Reason started looking after or giving special help to person 2: He/she wouldn't want anyone else caring for them"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reason started looking after or giving special help to person 2: Cared for person requested respondent's help/care"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reason started looking after or giving special help to person 2: Took over caring responsibilities from someone else"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reason started looking after or giving special help to person 2: They needed help (person cared for)"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Reason started looking after or giving special help to person 2: Other (SPECIFY)"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Assistance/ caring for person 2 affected ability to spend time doing leisure or social activities: Unable to socialise"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Assistance/ caring for person 2 affected ability to spend time doing leisure or social activities: Reduced time with spouse or partner"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Assistance/ caring for person 2 affected ability to spend time doing leisure or social activities: Reduced time with other family members"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Assistance/ caring for person 2 affected ability to spend time doing leisure or social activities: Reduced time with friends"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Assistance/ caring for person 2 affected ability to spend time doing leisure or social activities: Difficulties making new friends"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Assistance/ caring for person 2 affected ability to spend time doing leisure or social activities: Reduced time spent doing sport or physical activity"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Assistance/ caring for person 2 affected ability to spend time doing leisure or social activities: Reduced time spent doing pastime or hobby"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Assistance/ caring for person 2 affected ability to spend time doing leisure or social activities: Other"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Assistance/ caring for person 2 affected ability to spend time doing leisure or social activities: None of these"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 3: Last month helped with: Personal care"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 3: Last month helped with: Dealing with care services and benefits"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 3: Last month helped with: Physical help"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 3: Last month helped with: Other paperwork or financial benefits"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 3: Last month helped with: Other practical help (e.g. shopping for food)"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3: Last month helped with: Keeping him/her company"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 3: Last month helped with: Taking him/her out"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3: Last month helped with: Giving medicines"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3: Last month helped with: Keeping an eye on him/her to see he/she is alright"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3: Last month helped with: None of these"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 3: How often helped with personal care"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3: How often helped with physical help"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3: How often helped with dealing with care services and benefits"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3: How often helped with other paperwork or financial matters"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 3: How often helped with other practical help"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3: How often helped with keeping him/her company"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3: How often helped with taking him/her out"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3: How often helped with giving medicines"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 3: How often helped with keeping an eye on him/her to see if he/she is alright"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        # data_frame["Person 3: About how long been looking after or helping person 3"].replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 3 cared for receives regular visits at least once a month from: Doctor"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3 cared for receives regular visits at least once a month from: Community/ district nurse/ Community Matron"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 3 cared for receives regular visits at least once a month from: Health visitor"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3 cared for receives regular visits at least once a month from: Social worker/ care manager"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3 cared for receives regular visits at least once a month from: Home help/ care worker"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3 cared for receives regular visits at least once a month from: Meals on wheels"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 3 cared for receives regular visits at least once a month from: Voluntary worker"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3 cared for receives regular visits at least once a month from: Occupational therapist"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3 cared for receives regular visits at least once a month from: Educational professional"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3 cared for receives regular visits at least once a month from: Specialist/nursing care/ palliative care"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 3 cared for receives regular visits at least once a month from: Community mental health services"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3 cared for receives regular visits at least once a month from: Gardener/ caretaker/ warden"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3 cared for receives regular visits at least once a month from: Other professional visitor"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3 cared for receives regular visits at least once a month from: None"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 3 cared for has regular contact from: Doctor"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3 cared for has regular contact from: Community/ district nurse/ community matron"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3 cared for has regular contact from: Health visitor"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3 cared for has regular contact from: Social worker/ care manager"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 3 cared for has regular contact from: Home help/ care worker"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3 cared for has regular contact from: Meals on wheels"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3 cared for has regular contact from: Voluntary worker"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3 cared for has regular contact from: Occupational therapist"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 3 cared for has regular contact from: Educational professional"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3 cared for has regular contact from: Specialist/nursing care/ palliative care"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3 cared for has regular contact from: Community mental health services"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3 cared for has regular contact from: Gardener/ caretaker/ warden"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 3 cared for has regular contact from: Other professional visitor"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Person 3 cared for has regular contact from: None"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reasons for person 3 not receiving regular visits at least once a month: Not available/not offered"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reasons for person 3 not receiving regular visits at least once a month: Not needed"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Reasons for person 3 not receiving regular visits at least once a month: Not wanted by respondent"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reasons for person 3 not receiving regular visits at least once a month: Not wanted by the person respondent cares for"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reasons for person 3 not receiving regular visits at least once a month: Not at a convenient time"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Reasons for person 3 not receiving regular visits at least once a month: Too expensive"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reasons for person 3 not receiving regular visits at least once a month: Not eligible"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reasons for person 3 not receiving regular visits at least once a month: Don't know who to ask"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reasons for person 3 not receiving regular visits at least once a month: Other"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Person 3 cared for regularly makes use of a community or voluntary transport scheme"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reason started looking after or giving special help to person 3: No one else available"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reason started looking after or giving special help to person 3: Was willing / wanted to help out"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reason started looking after or giving special help to person 3: Had the time because was not working"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Reason started looking after or giving special help to person 3: Had the time because was working part time"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reason started looking after or giving special help to person 3: Have particular skills / ability to care"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reason started looking after or giving special help to person 3: Social Services (Local Authority) suggested respondent should provide care"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reason started looking after or giving special help to person 3: It was expected of respondent (it's what families do)"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Reason started looking after or giving special help to person 3: He/she wouldn't want anyone else caring for them"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reason started looking after or giving special help to person 3: Cared for person requested respondent's help/care"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reason started looking after or giving special help to person 3: Took over caring responsibilities from someone else"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Reason started looking after or giving special help to person 3: They needed help (person cared for)"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Reason started looking after or giving special help to person 3: Other (SPECIFY)"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Assistance/ caring for person 3 affected ability to spend time doing leisure or social activities: Unable to socialise"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Assistance/ caring for person 3 affected ability to spend time doing leisure or social activities: Reduced time with spouse or partner"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Assistance/ caring for person 3 affected ability to spend time doing leisure or social activities: Reduced time with other family members"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Assistance/ caring for person 3 affected ability to spend time doing leisure or social activities: Reduced time with friends"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Assistance/ caring for person 3 affected ability to spend time doing leisure or social activities: Difficulties making new friends"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Assistance/ caring for person 3 affected ability to spend time doing leisure or social activities: Reduced time spent doing sport or physical activity"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Assistance/ caring for person 3 affected ability to spend time doing leisure or social activities: Reduced time spent doing pastime or hobby"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Assistance/ caring for person 3 affected ability to spend time doing leisure or social activities: Other"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Assistance/ caring for person 3 affected ability to spend time doing leisure or social activities: None of these"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Seen or heard of Good Thinking"] \
            .replace('Not applicable', 'No - Not seen', inplace=True)

        data_frame[
            "How many times before HSE interview, looked at Every Mind Matters videos or information online"] \
            .replace('Not applicable', 'Never (SPONTANEOUS)', inplace=True)

        data_frame[
            "How many times before HSE interview, used Good Thinking online"] \
            .replace('Not applicable', 'Never', inplace=True)

        data_frame[
            '(D) 20+ hours provided (for care recipient for whom most hours provided)'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            '(D) 10+ hours provided (for care recipient for whom most hours provided)'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            '(D) Grouped hours provided 20+ (for care recipient for whom most hours provided)'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            '(D) Care recipient for most hours provided'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Any interest in taking up paid employment'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Which would help take up paid employment: The ability to work from home'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Which would help take up paid employment: Having some flexibility in the hours respondent wants to work'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Which would help take up paid employment: Access to affordable childcare'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Which would help take up paid employment: Access to affordable care for the person respondent cares for'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Which would help take up paid employment: Better public transport'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Which would help take up paid employment: Other'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Agree or disagree about potential barriers you might face in taking up paid employment: I cannot work because of my disability or health condition'] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Agree or disagree about potential barriers you might face in taking up paid employment: I cannot work because of my caring responsibilities'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Agree or disagree about potential barriers you might face in taking up paid employment: I am not sure I would be able to work regularly'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            'Agree or disagree about potential barriers you might face in taking up paid employment: At my age it is unlikely that I would find a suitable job'] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Agree or disagree about potential barriers you might face in taking up paid employment: I don't feel confident about working"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "Agree or disagree about potential barriers you might face in taking up paid employment: I haven't got enough qualifications and experience to find the right work"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Agree or disagree about potential barriers you might face in taking up paid employment: There aren't enough suitable job opportunities locally"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Agree or disagree about potential barriers you might face in taking up paid employment: I'm not sure I would be better off in work than on benefits"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Agree or disagree about potential barriers you might face in taking up paid employment: I cannot work because of my childcare responsibilities"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Agree or disagree about potential barriers you might face in taking up paid employment: My family/dependent(s) don't want me to work"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Aware that if you look after or give special help to sick, disabled or elderly people and have worked for the same employer for at least 26 weeks, now legally entitled to request flexible working"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "Made a request to work flexibly"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "(D) Type of help provided: Personal Care (e.g. washing self, having a bath or shower)"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "(D) Type of help provided: Physical help (e.g. to get around indoors, to get up and down stairs)"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "(D) Type of help provided: Help with dealing with care services and benefits"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "(D) Type of help provided: Help with other paperwork or financial matters"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "(D) Type of help provided: Other practical help (e.g. shopping for food)"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "(D) Type of help provided: Keeping them company"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "(D) Type of help provided: Taking them out"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "(D) Type of help provided: Giving medicines"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "(D) Type of help provided: Keeping an eye on them to see if they are alright"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "(D) Type of help provided: None of these"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "(D) How long care has been provided for (all people cared for, longest amount of time)"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "(D) How long care has been provided for (all people cared for, longest amount of time) (grouped)"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "(D) Impact of caring responsibilities on social contacts and leisure activities: unable to socialise"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "(D) Impact of caring responsibilities on social contacts and leisure activities: reduced time with spouse or partner"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "(D) Impact of caring responsibilities on social contacts and leisure activities: reduced time with other family members"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "(D) Impact of caring responsibilities on social contacts and leisure activities: reduced time with friends"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "(D) Impact of caring responsibilities on social contacts and leisure activities: difficulties making new friends"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "(D) Impact of caring responsibilities on social contacts and leisure activities: reduced time doing sport or physical activity"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "(D) Impact of caring responsibilities on social contacts and leisure activities: reduced time doing pastime or hobby"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            "(D) Impact of caring responsibilities on social contacts and leisure activities: other"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "(D) Impact of caring responsibilities on social contacts and leisure activities: none of these"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "(D) Awareness of flexible working rights for carers - carers who are currently employed"] \
            .replace('Not applicable', 'Not carer', inplace=True)
        data_frame[
            "(D) Requested flexible working - carers who are currently employed"] \
            .replace('Not applicable', 'Not carer', inplace=True)

        data_frame[
            'Received help in last month: Washing face and hands'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            'Received help in last month: Getting in and out of bed'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            'Received help in last month: Having a bath or a shower'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            'Received help in last month: Dressing or undressing, including putting on shoes and socks'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            'Received help in last month: Using the toilet'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            'Received help in last month: Eating, including cutting up food'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            'Received help in last month: Taking the right amount of medicine at the right times'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            'Received help in last month: Getting around indoors'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            'Received help in last month: Getting up and down stairs'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            'Received help in last month: Getting out of the house'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            'Received help in last month: Shopping for food'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            'Received help in last month: Doing routine housework or laundry'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            'Received help in last month: Doing paperwork or paying bills'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            'Whether received help because of health, disability or age problems'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            'Informal help provided: Husband/wife/partner'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            'Informal help provided: Son'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            'Informal help provided: Daughter'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            'Informal help provided: Grandchild'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            'Informal help provided: Brother/sister'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            'Informal help provided: Niece/nephew'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            'Informal help provided: Mother/father'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            'Informal help provided: Other family member'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            'Informal help provided: Friend'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            'Informal help provided: Neighbour'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            'Informal help provided: None of these'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            'Formal help provided: Home care helper/home help'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            'Formal help provided: A member of the reablement team'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            'Formal help provided: Occupational Therapist/ physiotherapist'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            'Formal help provided: Voluntary helper'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            'Formal help provided: Warden/Sheltered housing'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            'Formal help provided: Cleaner'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            "Formal help provided: Council's handyman"] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            'Formal help provided: Other - please specify'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            'Formal help provided: None of these'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) Unmet need for any personal activities (ADLs)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) Whether any unmet need for any personal activities (ADLs)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) Unmet need for any instrumental activities (IADLs)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) Whether any unmet need for any instrumental activities (IADLs)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) Spouse/partner helped with ADLs (tasks A-I)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) Son helped with ADLs (tasks A-I)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) Daughter helped with ADLs (tasks A-I)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) Friend/Neighbour helped with ADLs (tasks A-I)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) Other member of the family helped with ADLs (tasks A-I)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) No informal helpers helped with ADLs (tasks A-I)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) Any informal helper helped with ADLs (tasks A-I)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) No informal helper helped with IADLs (tasks J-M)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Any informal helper helped with IADLs (tasks J-M)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Home care worker helped with ADLs (tasks A-I)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Other formal helper helped with ADLs (tasks A-I)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) No formal helpers helped with ADLs (tasks A-I)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Any formal helper helped with ADL tasks (A-I)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Home care worker helped with IADLs (tasks J-M)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Other formal helper helped with IADLs (tasks J-M)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) No formal helpers helped with IADLs (tasks J-M)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Any formal helper helped with IADL tasks (J-M)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Who provided IADL help (informal/formal helpers, tasks J-M)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Grouped spouse hours who helped (6 groups, 50+)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Grouped spouse hours who helped (4 groups, 20+)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Grouped spouse hours who helped (4 groups, 10+)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Grouped, hours of help provided in the last week by the son who helped the most (9 groups)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Grouped, hours of help provided in the last week by son who helped the most (6 groups, 50+)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Grouped, hours of help provided in the last week by son who helped the most (4 groups, 10+)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Grouped, hours of help provided in the last week by son who helped the most (4 groups, 20+)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Grouped, hours of help provided in the last week by daughter who helped the most (9 groups)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Grouped, hours of help provided in the last week by daughter who helped the most (6 groups, 50+)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Grouped, hours of help provided in the last week by daughter who helped the most (4 groups, 10+)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Grouped, hours of help provided in the last week by daughter who helped the most (4 groups, 20+)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Grouped, hours of help provided in the last week by other family member who helped the most (9 groups)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Other family member who provided most hours of care'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Grouped, hours of help provided in the last week by other family member who helped the most (6 groups, 50+)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Grouped, hours of help provided in the last week by other family member who helped the most (4 groups, 10+)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Grouped, hours of help provided in the last week by other family member who helped the most (4 groups, 20+)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Grouped, hours of help provided in the last week by friend or neighbour who helped the most (9 groups)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Grouped, hours of help provided in the last week by friend or neighbour who helped the most (6 groups, 50+)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Grouped, hours of help provided in the last week by friend or neighbour who helped the most (4 groups, 10+)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Grouped, hours of help provided in the last week by friend or neighbour who helped the most (4 groups, 20+)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Hours of help provided in the last week by home care worker who helped the most'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Grouped, hours of help provided in the last week by home care worker who helped the most (9 groups)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Grouped, hours of help provided in the last week by home care worker who helped the most (6 groups, 50+)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Grouped, hours of help provided in the last week by home care worker who helped the most (4 groups, 10+)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame[
            '(D) Grouped, hours of help provided in the last week by home care worker who helped the most (4 groups, 20+)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) Spouse/partner helped with IADLs (tasks J-M)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) Son helped with IADLs (tasks J-M)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) Daughter helped with IADLs (tasks J-M)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) Friend/neighbour helped with IADLs (tasks J-M)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) Other family member helped with IADLs (tasks J-M)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame[
            '(D) Did you receive help: Stairs (TASK I)'] \
            .replace('Not applicable', 'No help required for any task', inplace=True)

        data_frame[
            '(D) Did you receive help: Indoors (TASK H)'] \
            .replace('Not applicable', 'No help required for any task', inplace=True)

        data_frame[
            '(D) Did you receive help: Bed (TASK A)'] \
            .replace('Not applicable', 'No help required for any task', inplace=True)

        data_frame[
            '(D) Did you receive help: Shower (TASK C)'] \
            .replace('Not applicable', 'No help required for any task', inplace=True)

        data_frame[
            '(D) Did you receive help: Wash (TASK B)'] \
            .replace('Not applicable', 'No help required for any task', inplace=True)

        data_frame[
            '(D) Did you receive help: Dress (TASK D)'] \
            .replace('Not applicable', 'No help required for any task', inplace=True)

        data_frame[
            '(D) Did you receive help: Medicine (TASK G)'] \
            .replace('Not applicable', 'No help required for any task', inplace=True)

        data_frame[
            '(D) Did you receive help: Eat (TASK F)'] \
            .replace('Not applicable', 'No help required for any task', inplace=True)

        data_frame[
            '(D) Did you receive help: House (TASK J)'] \
            .replace('Not applicable', 'No help required for any task', inplace=True)

        data_frame[
            '(D) Did you receive help: Toilet (TASK E)'] \
            .replace('Not applicable', 'No help required for any task', inplace=True)

        data_frame[
            '(D) Did you receive help: Housework (TASK L)'] \
            .replace('Not applicable', 'No help required for any task', inplace=True)

        data_frame[
            '(D) Did you receive help: Paperwork (TASK M)'] \
            .replace('Not applicable', 'No help required for any task', inplace=True)

        data_frame[
            '(D) Did you receive help: Shop (TASK K)'] \
            .replace('Not applicable', 'No help required for any task', inplace=True)

        data_frame[
            '(D) Whether any unmet ADL and/or IADL needs'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame['Whether need help: Getting in and out of bed'] \
            .replace('Not applicable', 'I can do this without help from anyone', inplace=True)

        data_frame['Whether need help: Washing face and hands'] \
            .replace('Not applicable', 'I can do this without help from anyone', inplace=True)

        data_frame['Whether need help: Having a bath/shower, including getting in and out of bath/shower'] \
            .replace('Not applicable', 'I can do this without help from anyone', inplace=True)

        data_frame['Whether need help: Dressing and undressing, including putting on shoes and socks'] \
            .replace('Not applicable', 'I can do this without help from anyone', inplace=True)

        data_frame['Whether need help: Using the toilet'] \
            .replace('Not applicable', 'I can do this without help from anyone', inplace=True)

        data_frame['Whether need help: Eating, including cutting up food'] \
            .replace('Not applicable', 'I can do this without help from anyone', inplace=True)

        data_frame['Whether need help: Taking the right amount of medicine at the right times'] \
            .replace('Not applicable', 'I can do this without help from anyone', inplace=True)

        data_frame['Whether need help: Getting around indoors'] \
            .replace('Not applicable', 'I can do this without help from anyone', inplace=True)

        data_frame['Whether need help: Getting up and down stairs'] \
            .replace('Not applicable', 'I can do this without help from anyone', inplace=True)

        data_frame['Whether need help: Getting out of the house'] \
            .replace('Not applicable', 'I can do this without help from anyone', inplace=True)

        data_frame['Whether need help: Shopping for food'] \
            .replace('Not applicable', 'I can do this without help from anyone', inplace=True)

        data_frame['Whether need help: Doing routine housework or laundry'] \
            .replace('Not applicable', 'I can do this without help from anyone', inplace=True)

        data_frame['Whether need help: Doing paperwork or paying bills'] \
            .replace('Not applicable', 'I can do this without help from anyone', inplace=True)

        data_frame['Health affected in the last three months by help/ support provided: Feeling depressed'] \
            .replace('Not applicable', 'Not Carer', inplace=True)

        data_frame['Health affected in the last three months by help/ support provided: Loss of appetite'] \
            .replace('Not applicable', 'Not Carer', inplace=True)

        data_frame['Health affected in the last three months by help/ support provided: Disturbed sleep'] \
            .replace('Not applicable', 'Not Carer', inplace=True)

        data_frame['Health affected in the last three months by help/ support provided: General feeling of stress'] \
            .replace('Not applicable', 'Not Carer', inplace=True)

        data_frame['Health affected in the last three months by help/ support provided: Physical strain'] \
            .replace('Not applicable', 'Not Carer', inplace=True)

        data_frame['Health affected in the last three months by help/ support provided: Short tempered'] \
            .replace('Not applicable', 'Not Carer', inplace=True)

        data_frame[
            'Health affected in the last three months by help/ support provided: Developed my own health condition'] \
            .replace('Not applicable', 'Not Carer', inplace=True)

        data_frame[
            'Health affected in the last three months by help/ support provided: Made an existing condition worse'] \
            .replace('Not applicable', 'Not Carer', inplace=True)

        data_frame['Health affected in the last three months by help/ support provided: Other'] \
            .replace('Not applicable', 'Not Carer', inplace=True)

        data_frame['Health affected in the last three months by help/ support provided: None of these'] \
            .replace('Not applicable', 'Not Carer', inplace=True)

        data_frame['Seen GP because health has been affected by the support given'] \
            .replace('Not applicable', 'Not Carer', inplace=True)

        data_frame[
            'Local Authority (council) carried out a carer’s assessment as a result of the help or support provide'] \
            .replace('Not applicable', 'Not Carer', inplace=True)

        data_frame['Last 12 months, has caring caused any financial difficulties'] \
            .replace('Not applicable', 'Not Carer', inplace=True)

        data_frame[
            'Thinking about combining your paid work and caring responsibilities, which of the following statements best describes your current situation'] \
            .replace('Not applicable', 'Not Carer', inplace=True)

        data_frame['Whether needed any help with tasks'].replace('Not applicable', 'No', inplace=True)

        data_frame['(D) Need help (binary): Bed (TASK A)'].replace('Not applicable', 'No help needed',
                                                                   inplace=True)
        data_frame['(D) Need help (binary): Wash (TASK B)'].replace('Not applicable', 'No help needed',
                                                                    inplace=True)
        data_frame['(D) Need help (binary): Shower (TASK C)'].replace('Not applicable', 'No help needed',
                                                                      inplace=True)
        data_frame['(D) Need help (binary): Dress (TASK D)'].replace('Not applicable', 'No help needed',
                                                                     inplace=True)
        data_frame['(D) Need help (binary): Toilet (TASK E)'].replace('Not applicable', 'No help needed',
                                                                      inplace=True)
        data_frame['(D) Need help (binary): Eat (TASK F)'].replace('Not applicable', 'No help needed',
                                                                   inplace=True)
        data_frame['(D) Need help (binary): Medicine (TASK G)'].replace('Not applicable', 'No help needed',
                                                                        inplace=True)
        data_frame['(D) Need help (binary): Indoors (TASK H)'].replace('Not applicable', 'No help needed',
                                                                       inplace=True)
        data_frame['(D) Need help (binary): Stairs (TASK I)'].replace('Not applicable', 'No help needed',
                                                                      inplace=True)
        data_frame['(D) Need help (binary): House (TASK J)'].replace('Not applicable', 'No help needed',
                                                                     inplace=True)
        data_frame['(D) Need help (binary): Shop (TASK K)'].replace('Not applicable', 'No help needed',
                                                                    inplace=True)
        data_frame['(D) Need help (binary): Housework (TASK L)'].replace('Not applicable', 'No help needed',
                                                                         inplace=True)
        data_frame['(D) Need help (binary): Paperwork (TASK M)'].replace('Not applicable', 'No help needed',
                                                                         inplace=True)
        data_frame['Help/support given because of health/old age or help more generally'] \
            .replace('Not applicable', 'Not providing support', inplace=True)

        data_frame['Informal help for bath: Husband/wife/partner'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Informal help for bath: Son'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Informal help for bath: Daughter'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Informal help for bath: Grandchild'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Informal help for bath: Brother/sister'] \
            .replace('Not applicable', 'No support needed', inplace=True)

        data_frame['Informal help for bath: Niece/nephew'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Informal help for bath: Mother/father'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Informal help for bath: Other family member'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Informal help for bath: Friend'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Informal help for bath: Neighbour'] \
            .replace('Not applicable', 'No support needed', inplace=True)

        data_frame['Informal help for bath: None of the above'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Formal help for bath: Home care worker/home help'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Formal help for bath: A member of the reablement team'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Formal help for bath: Occupational Therapist/physiotherapist'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Formal help for bath: Voluntary helper'] \
            .replace('Not applicable', 'No support needed', inplace=True)

        data_frame['Formal help for bath: Warden/Sheltered housing'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Formal help for bath: Cleaner'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame["Formal help for bath: Council's handyman"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Formal help for bath: Other'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Formal help for bath: None of the above'] \
            .replace('Not applicable', 'No support needed', inplace=True)

        data_frame['Informal help for basic indoor tasks: Husband/wife/partner'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Informal help for basic indoor tasks: Son'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame["Informal help for basic indoor tasks: Daughter"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Informal help for basic indoor tasks: Grandchild'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Informal help for basic indoor tasks: Brother/sister'] \
            .replace('Not applicable', 'No support needed', inplace=True)

        data_frame['Informal help for basic indoor tasks: Niece/nephew'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Informal help for basic indoor tasks: Mother/father'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame["Informal help for basic indoor tasks: Other family member"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Informal help for basic indoor tasks: Friend'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Informal help for basic indoor tasks: Neighbour'] \
            .replace('Not applicable', 'No support needed', inplace=True)

        data_frame['Informal help for basic indoor tasks: None of the above'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Formal help for basic indoor tasks: Home care worker/home help'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame["Formal help for basic indoor tasks: A member of the reablement team"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Formal help for basic indoor tasks: Occupational Therapist/physiotherapist'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Formal help for basic indoor tasks: Voluntary helper'] \
            .replace('Not applicable', 'No support needed', inplace=True)

        data_frame['Formal help for basic indoor tasks: Warden/Sheltered housing'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Formal help for basic indoor tasks: Cleaner'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame["Formal help for basic indoor tasks: Council's handyman"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Formal help for basic indoor tasks: Other'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Formal help for basic indoor tasks: None of the above'] \
            .replace('Not applicable', 'No support needed', inplace=True)

        data_frame['Informal help for outdoor tasks & housework: Husband/wife/partner'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Informal help for outdoor tasks & housework: Son'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame["Informal help for outdoor tasks & housework: Daughter"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Informal help for outdoor tasks & housework: Grandchild'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Informal help for outdoor tasks & housework: Brother/sister'] \
            .replace('Not applicable', 'No support needed', inplace=True)

        data_frame['Informal help for outdoor tasks & housework: Niece/nephew'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Informal help for outdoor tasks & housework: Mother/father'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame["Informal help for outdoor tasks & housework: Other family member"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Informal help for outdoor tasks & housework: Friend'] \
            .replace('Not applicable', 'No support needed', inplace=True)

        data_frame['Informal help for outdoor tasks & housework: Neighbour'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Informal help for outdoor tasks & housework: None of the above'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame["Formal help for outdoor tasks & housework: Home care worker/home help"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Formal help for outdoor tasks & housework: A member of the reablement team'] \
            .replace('Not applicable', 'No support needed', inplace=True)

        data_frame['Formal help for outdoor tasks & housework: Occupational Therapist/physiotherapist'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Formal help for outdoor tasks & housework: Voluntary helper'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame["Formal help for outdoor tasks & housework: Warden/Sheltered housing"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Formal help for outdoor tasks & housework: Cleaner'] \
            .replace('Not applicable', 'No support needed', inplace=True)

        data_frame["Formal help for outdoor tasks & housework: Council's handyman"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Formal help for outdoor tasks & housework: Other'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame["Formal help for outdoor tasks & housework: None of the above"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Spouse/partner: Person lives in household'] \
            .replace('Not applicable', 'No support needed', inplace=True)

        data_frame["Son: Person lives in household"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['2nd Son: Person lives in household'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame["Daughter: Person lives in household"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['2nd Daughter: Person lives in household'] \
            .replace('Not applicable', 'No support needed', inplace=True)

        data_frame["3rd Daughter: Person lives in household"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame["3rd Son: Person lives in household"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Grandchild: Person lives in household'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame["2nd Grandchild: Person lives in household"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Brother/sister: Person lives in household'] \
            .replace('Not applicable', 'No support needed', inplace=True)

        data_frame["2nd Brother/sister: Person lives in household"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['3rd Brother/sister: Person lives in household'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame["Niece/nephew: Person lives in household"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['2nd Niece/nephew: Person lives in household'] \
            .replace('Not applicable', 'No support needed', inplace=True)

        data_frame["Other family member: Person lives in household"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Friend: Person lives in household'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame["2nd Friend: Person lives in household"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['3rd Friend: Person lives in household'] \
            .replace('Not applicable', 'No support needed', inplace=True)

        data_frame["Hours of formal help received in last week: Home care worker/home help/personal assistant"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Hours of formal help received in last week: Member of the reablement/intermediate care staff team'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame["Hours of formal help received in last week: Warden/sheltered housing manager"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Hours of formal help received in last week: Cleaner'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Hours of formal help received in last week: Occupational therapist/physiotherapist'] \
            .replace('Not applicable', 'No support needed', inplace=True)

        data_frame["Hours of formal help received in last week: Other"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Grouped hours of informal help received in last week (9 groups): Husband/wife/partner'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame["Grouped hours of informal help received in last week (9 groups): Son"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Grouped hours of informal help received in last week (9 groups): Daughter'] \
            .replace('Not applicable', 'No support needed', inplace=True)

        data_frame["Grouped hours of informal help received in last week (9 groups): Grandchild"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Grouped hours of informal help received in last week (9 groups): Brother/sister'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame["Grouped hours of informal help received in last week (9 groups): Niece/nephew"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Grouped hours of informal help received in last week (9 groups): Other family member'] \
            .replace('Not applicable', 'No support needed', inplace=True)

        data_frame["Grouped hours of informal help received in last week (9 groups): Friend"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Average hours of help in a usual week: Son'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame["Average hours of help in a usual week: Daughter"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Average hours of help in a usual week: Friend'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Average hours of help in a usual week: Grandchild'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Average hours of help in a usual week: Brother/sister'] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame["Amount of time been receiving these kinds of help"] \
            .replace('Not applicable', 'No support needed', inplace=True)
        data_frame['Unpaid care received'] \
            .replace('Not applicable', 'No support needed', inplace=True)

        data_frame["Whether respondent answered on own"] \
            .replace('Not applicable', 'nether', inplace=True)
        data_frame['Average hours of help in a usual week: Friend'] \
            .replace('Not applicable', 'No support needed', inplace=True)

        data_frame['(D) Unmet need: Stairs (TASK I)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame['(D) Unmet need: Indoors (TASK H)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame["(D) Unmet need: Bed (TASK A)"] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame['(D) Unmet need: Shower (TASK C)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame['(D) Unmet need: Dress (TASK D)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame['(D) Unmet need: Wash (TASK B)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame["(D) Unmet need: Toilet (TASK E)"] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame['(D) Unmet need: Medicine (TASK G)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame['(D) Unmet need: Eat (TASK F)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame['(D) Unmet need: House (TASK J)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame["(D) Unmet need: Shop (TASK K)"] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame['(D) Unmet need: Housework (TASK L)'] \
            .replace('Not applicable', 'No help needed', inplace=True)
        data_frame['(D) Unmet need: Paperwork/Bills (TASK M)'] \
            .replace('Not applicable', 'No help needed', inplace=True)

        data_frame.drop(index=data_frame[data_frame["(D) Number provided help to - grouped"] == "Don't know"].index,
                        inplace=True)
        data_frame['Number of people provide this type of help and support to'] \
            .replace('Not applicable', 0, inplace=True)
        data_frame['Number of people provide this type of help and support to'] \
            .replace("Don't know", 0, inplace=True)
        data_frame['Number of people provide this type of help and support to'] = pd.to_numeric(
            data_frame['Number of people provide this type of help and support to'], errors='coerce')

        data_frame['Number of people provide this type of help and support to'] \
            .replace('Not applicable', 0, inplace=True)
        data_frame['Number of people provide this type of help and support to'] = pd.to_numeric(
            data_frame['Number of people provide this type of help and support to'], errors='coerce')

        data_frame['Ability to take up or stay in employment affected: Left employment altogether'] \
            .replace('Not applicable', 'Not Carer', inplace=True)

        data_frame['Ability to take up or stay in employment affected: Took new job'] \
            .replace('Not applicable', 'Not Carer', inplace=True)

        data_frame['Ability to take up or stay in employment affected: Worked fewer hours'] \
            .replace('Not applicable', 'Not Carer', inplace=True)

        data_frame['Ability to take up or stay in employment affected: Reduced responsibility at work'] \
            .replace('Not applicable', 'Not Carer', inplace=True)

        data_frame['Ability to take up or stay in employment affected: Flexible employment agreed'] \
            .replace('Not applicable', 'Not Carer', inplace=True)

        data_frame['Ability to take up or stay in employment affected: Changed to work at home'] \
            .replace('Not applicable', 'Not Carer', inplace=True)

        data_frame['Ability to take up or stay in employment affected: Other'] \
            .replace('Not applicable', 'Not Carer', inplace=True)

        data_frame['Ability to take up or stay in employment affected: Employment not affected'] \
            .replace('Not applicable', 'Not Carer', inplace=True)

        # remove height & weight notes
        data_frame.drop(
            index=data_frame[data_frame["(D) Final height - measured or estimated (cm)"] == "Not applicable"].index,
            inplace=True)
        data_frame.drop(
            index=data_frame[data_frame["(D) Final weight - measured or estimated (kg)"] == "Not applicable"].index,
            inplace=True)

        # remove Waist circumference
        data_frame.drop(index=data_frame[data_frame["(D) Valid Mean Waist (cm)"] == "Not applicable"].index,
                        inplace=True)
        data_frame['(D) Valid Mean Waist (cm)'] = pd.to_numeric(data_frame['(D) Valid Mean Waist (cm)'],
                                                                errors='coerce')

        data_frame.drop(index=data_frame[data_frame["(D) Valid Mean Hip (cm)"] == "Not applicable"].index, inplace=True)
        data_frame['(D) Valid Mean Hip (cm)'] = pd.to_numeric(data_frame['(D) Valid Mean Hip (cm)'], errors='coerce')

        DataController.convert_by_median(data_frame, '(D) Omron Diastolic BP (mean 2nd/3rd) inc. invalid')
        DataController.convert_by_median(data_frame, '(D) Omron Systolic BP (mean 2nd/3rd) inc. invalid')
        DataController.convert_by_median(data_frame, '(D) Omron Mean arterial pressure (mean 2nd/3rd) inc. invalid')

        # Cotinine
        data_frame["(D) Binary of undetectable cotinine, <0.1ng/ml (16+yrs, excl users NDP)"].replace('Not applicable',
                                                                                                      'Refused',
                                                                                                      inplace=True)
        data_frame["(D) Binary of valid cotinine levels at 12+ ng/ml (16+, incl users of NDP)"].replace(
            'Not applicable', 'Refused', inplace=True)
        data_frame["(D) Binary of valid cotinine levels at 12+ ng/ml (16+yrs, excl users of NDP)"].replace(
            'Not applicable', 'Refused', inplace=True)
        data_frame['Cotinine result'].replace('Not applicable', 0, inplace=True)

        # drop age columns into only one
        data_frame.drop(columns=['(D) Age 16+ in ten year bands', '(D) Age 16+, 5 year bands',
                                 '(D) Age 2-15 in three groups'], inplace=True)
        data_frame['(D) Age, 3 year bands for 0-15, 5 year bands 16+'].replace('90+', '90-100', inplace=True)
        data_frame['age mean'] = data_frame['(D) Age, 3 year bands for 0-15, 5 year bands 16+'].apply(
            lambda x: DataController.split_mean(x))
        data_frame['age mean'].replace(0.5, 1, inplace=True)
        data_frame.rename(columns={'(D) Age, 3 year bands for 0-15, 5 year bands 16+': 'age_mean'}, inplace=True)
        data_frame['age_mean'] = data_frame['age mean']
        data_frame.drop(columns=['age mean'], inplace=True)

        # drop under 18s
        data_frame['age_mean'] = pd.to_numeric(data_frame['age_mean'], errors='coerce')
        data_frame.drop(index=data_frame[data_frame["age_mean"] <= 18].index, inplace=True)

        # units of drink
        data_frame['Amount normal beer (small cans/bottles) on heaviest day (capi+casi)'].replace("Don't know", 0,
                                                                                                  inplace=True)
        data_frame['Amount normal beer (small cans/bottles) on heaviest day (capi+casi)'].replace("Not applicable", 0,
                                                                                                  inplace=True)
        data_frame['Amount normal beer (small cans/bottles) on heaviest day (capi+casi)'] = \
            pd.to_numeric(data_frame['Amount normal beer (small cans/bottles) on heaviest day (capi+casi)'],
                          errors='coerce')

        data_frame['Normal beer bottle size (pints) on heaviest day (capi+casi)'].replace("Not applicable", 0,
                                                                                          inplace=True)
        data_frame['Normal beer bottle size (pints) on heaviest day (capi+casi)'] = \
            pd.to_numeric(data_frame['Normal beer bottle size (pints) on heaviest day (capi+casi)'], errors='coerce')

        data_frame['Amount normal beer (pints) on heaviest day (capi+casi)'].replace("Don't know", 0,
                                                                                     inplace=True)
        data_frame['Amount normal beer (pints) on heaviest day (capi+casi)'].replace("Not applicable", 0,
                                                                                     inplace=True)
        data_frame['Amount normal beer (pints) on heaviest day (capi+casi)'] = \
            pd.to_numeric(data_frame['Amount normal beer (pints) on heaviest day (capi+casi)'], errors='coerce')

        data_frame['Amount sherry (glasses) on heaviest day (capi+casi)'].replace("Not applicable", 0,
                                                                                  inplace=True)
        data_frame['Amount sherry (glasses) on heaviest day (capi+casi)'] = \
            pd.to_numeric(data_frame['Amount sherry (glasses) on heaviest day (capi+casi)'], errors='coerce')

        data_frame['Amount wine (250ml glasses) on heaviest day (capi+casi)'].replace("Not applicable", 0,
                                                                                      inplace=True)
        data_frame['Amount wine (250ml glasses) on heaviest day (capi+casi)'] = \
            pd.to_numeric(data_frame['Amount wine (250ml glasses) on heaviest day (capi+casi)'], errors='coerce')

        data_frame['Amount wine (175ml glasses) on heaviest day (capi+casi)'].replace("Not applicable", 0,
                                                                                      inplace=True)
        data_frame['Amount wine (175ml glasses) on heaviest day (capi+casi)'] = \
            pd.to_numeric(data_frame['Amount wine (175ml glasses) on heaviest day (capi+casi)'], errors='coerce')

        data_frame['Amount wine (125ml glasses) on heaviest day (capi+casi)'].replace("Not applicable", 0,
                                                                                      inplace=True)
        data_frame['Amount wine (125ml glasses) on heaviest day (capi+casi)'] = \
            pd.to_numeric(data_frame['Amount wine (125ml glasses) on heaviest day (capi+casi)'], errors='coerce')

        data_frame['Amount wine (125ml glasses from a bottle) on heaviest day (capi+casi)'].replace("Not applicable", 0,
                                                                                                    inplace=True)
        data_frame['Amount wine (125ml glasses from a bottle) on heaviest day (capi+casi)'] = \
            pd.to_numeric(data_frame['Amount wine (125ml glasses from a bottle) on heaviest day (capi+casi)'],
                          errors='coerce')

        data_frame['Amount alcopops (standard bottles) on heaviest day (capi+casi)'].replace("Not applicable", 0,
                                                                                             inplace=True)
        data_frame['Amount alcopops (standard bottles) on heaviest day (capi+casi)'].replace("Don't know", 0,
                                                                                             inplace=True)
        data_frame['Amount alcopops (standard bottles) on heaviest day (capi+casi)'] = \
            pd.to_numeric(data_frame['Amount alcopops (standard bottles) on heaviest day (capi+casi)'],
                          errors='coerce')

        data_frame['Amount alcopops (small cans) on heaviest day (capi+casi)'].replace("Not applicable", 0,
                                                                                       inplace=True)
        data_frame['Amount alcopops (small cans) on heaviest day (capi+casi)'] = \
            pd.to_numeric(data_frame['Amount alcopops (small cans) on heaviest day (capi+casi)'],
                          errors='coerce')

        data_frame['Amount of alcopops usually drunk on any one day (small cans) (Capi)'].replace("Not applicable", 0,
                                                                                                  inplace=True)
        data_frame['Amount of alcopops usually drunk on any one day (small cans) (Capi)'] = \
            pd.to_numeric(data_frame['Amount of alcopops usually drunk on any one day (small cans) (Capi)'],
                          errors='coerce')

        data_frame['Amount of alcopops usually drunk on any one day (large bottles) (Capi)'].replace("Not applicable",
                                                                                                     0, inplace=True)
        data_frame['Amount of alcopops usually drunk on any one day (large bottles) (Capi)'] = \
            pd.to_numeric(data_frame['Amount of alcopops usually drunk on any one day (large bottles) (Capi)'],
                          errors='coerce')

        data_frame['Amount of alcopops usually drunk on any one day (standard bottles) (Capi)'].replace(
            "Not applicable", 0, inplace=True)
        data_frame['Amount of alcopops usually drunk on any one day (standard bottles) (Capi)'] = \
            pd.to_numeric(data_frame['Amount of alcopops usually drunk on any one day (standard bottles) (Capi)'],
                          errors='coerce')

        data_frame['(D) Units of wine/week'].replace("Refused", 0, inplace=True)
        data_frame['(D) Units of wine/week'].replace("Don't know", 0, inplace=True)
        data_frame['(D) Units of wine/week'] = pd.to_numeric(data_frame['(D) Units of wine/week'], errors='coerce')

        data_frame['Amount of normal beer etc. usually drunk on any one day (bottles) (Capi)'].replace("Not applicable",
                                                                                                       0, inplace=True)
        data_frame['Amount of normal beer etc. usually drunk on any one day (bottles) (Capi)'] = pd.to_numeric(
            data_frame['Amount of normal beer etc. usually drunk on any one day (bottles) (Capi)'],
            errors='coerce')
        data_frame['Amount of normal beer etc. usually drunk on any one day (large cans) (Capi)'].replace(
            "Not applicable",
            0, inplace=True)
        data_frame['Amount of normal beer etc. usually drunk on any one day (large cans) (Capi)'] = pd.to_numeric(
            data_frame['Amount of normal beer etc. usually drunk on any one day (large cans) (Capi)'],
            errors='coerce')
        data_frame['Amount of normal beer etc. usually drunk on any one day (small cans) (Capi)'].replace(
            "Not applicable",
            0, inplace=True)
        data_frame['Amount of normal beer etc. usually drunk on any one day (small cans) (Capi)'] = pd.to_numeric(
            data_frame['Amount of normal beer etc. usually drunk on any one day (small cans) (Capi)'],
            errors='coerce')
        data_frame['Amount of normal beer etc. usually drunk on any one day (pints) (Capi)'].replace("Not applicable",
                                                                                                     0, inplace=True)
        data_frame['Amount of normal beer etc. usually drunk on any one day (pints) (Capi)'] = \
            pd.to_numeric(data_frame['Amount of normal beer etc. usually drunk on any one day (pints) (Capi)'],
                          errors='coerce')

        data_frame['Amount normal beer (large cans/bottles) on heaviest day (capi+casi)'].replace("Not applicable", 0,
                                                                                                  inplace=True)
        data_frame['Amount normal beer (large cans/bottles) on heaviest day (capi+casi)'] = \
            pd.to_numeric(data_frame['Amount normal beer (large cans/bottles) on heaviest day (capi+casi)'],
                          errors='coerce')

        data_frame['How many normal beer (bottles) on heaviest day (capi+casi)'].replace("Not applicable", 0,
                                                                                         inplace=True)
        data_frame['How many normal beer (bottles) on heaviest day (capi+casi)'] = \
            pd.to_numeric(data_frame['How many normal beer (bottles) on heaviest day (capi+casi)'],
                          errors='coerce')

        data_frame['(D) Units of normal beer/week'].replace("Don't know", 0, inplace=True)
        data_frame['(D) Units of normal beer/week'] = pd.to_numeric(data_frame['(D) Units of normal beer/week'],
                                                                    errors='coerce')
        data_frame['(D) Units of strong beer/week'].replace("Don't know", 0, inplace=True)
        data_frame['(D) Units of strong beer/week'] = pd.to_numeric(data_frame['(D) Units of strong beer/week'],
                                                                    errors='coerce')

        data_frame['Amount spirits (measures) on heaviest day (capi+casi)'].replace("Not applicable", 0, inplace=True)
        data_frame['Amount spirits (measures) on heaviest day (capi+casi)'].replace("Don't know", 0, inplace=True)
        data_frame['Amount spirits (measures) on heaviest day (capi+casi)'] = pd.to_numeric(
            data_frame['Amount spirits (measures) on heaviest day (capi+casi)'],
            errors='coerce')

        data_frame['(D) Units of spirits/week'].replace("Refused", 0, inplace=True)
        data_frame['(D) Units of spirits/week'].replace("Don't know", 0, inplace=True)
        data_frame['(D) Units of spirits/week'] = pd.to_numeric(data_frame['(D) Units of spirits/week'],
                                                                errors='coerce')
        data_frame['(D) Units of sherry/week'].replace("Don't know", 0, inplace=True)
        data_frame['(D) Units of sherry/week'] = pd.to_numeric(data_frame['(D) Units of sherry/week'], errors='coerce')
        data_frame['(D) Units of alcopops/week'].replace("Don't know", 0, inplace=True)
        data_frame['(D) Units of alcopops/week'] = pd.to_numeric(data_frame['(D) Units of alcopops/week'],
                                                                 errors='coerce')
        data_frame['How many days in last 7 had a drink (capi+casi)'].replace('Not applicable', 0, inplace=True)
        data_frame['How many days in last 7 had a drink (capi+casi)'].replace("Don't know", 0, inplace=True)
        data_frame['How many days in last 7 had a drink (capi+casi)'] = \
            pd.to_numeric(data_frame['How many days in last 7 had a drink (capi+casi)'], errors='coerce')
        data_frame['(D) Total units of alcohol/week'].replace("Refused", 0, inplace=True)
        data_frame['(D) Total units of alcohol/week'].replace("Don't know", 0, inplace=True)
        data_frame['(D) Total units of alcohol/week'] = pd.to_numeric(data_frame['(D) Total units of alcohol/week'],
                                                                      errors='coerce')
        data_frame['Which day drank the most in the last 7 days (Capi)'].replace("Not applicable", 'Have not drank',
                                                                                 inplace=True)
        data_frame['Whether drank more on a particular day in the last 7 days (Capi)'].replace("Not applicable",
                                                                                               'Have not drank',
                                                                                               inplace=True)
        data_frame['(D) Number of days drank in last week, including none'].replace("Don't know", 0, inplace=True)
        data_frame['(D) Number of days drank in last week, including none'].replace('Not applicable', 0, inplace=True)
        data_frame['(D) Number of days drank in last week, including none'] = pd.to_numeric(
            data_frame['(D) Number of days drank in last week, including none'], errors='coerce')
        data_frame['(D) Normal beer bottle multiplier (16yrs+)'] = pd.to_numeric(
            data_frame['(D) Normal beer bottle multiplier (16yrs+)'], errors='coerce')
        data_frame['Amount of wine usually drunk on any one day (Capi)'].replace('Not applicable', 0, inplace=True)
        data_frame['Amount of wine usually drunk on any one day (Capi)'].replace("Don't know", 0, inplace=True)
        data_frame['Amount of wine usually drunk on any one day (Capi)'] = pd.to_numeric(
            data_frame['Amount of wine usually drunk on any one day (Capi)'], errors='coerce')

        data_frame['Amount of strong beer etc. usually drunk on any one day (pints) (Capi)'].replace('Not applicable',
                                                                                                     0, inplace=True)
        data_frame['Amount of strong beer etc. usually drunk on any one day (pints) (Capi)'] = \
            pd.to_numeric(data_frame['Amount of strong beer etc. usually drunk on any one day (pints) (Capi)'],
                          errors='coerce')
        data_frame['Amount of strong beer etc. usually drunk on any one day (small cans) (Capi)'].replace(
            'Not applicable', 0, inplace=True)
        data_frame['Amount of strong beer etc. usually drunk on any one day (small cans) (Capi)'] = \
            pd.to_numeric(data_frame['Amount of strong beer etc. usually drunk on any one day (small cans) (Capi)'],
                          errors='coerce')
        data_frame['Amount of strong beer etc. usually drunk on any one day (large cans) (Capi)'].replace(
            'Not applicable', 0, inplace=True)
        data_frame['Amount of strong beer etc. usually drunk on any one day (large cans) (Capi)'] = \
            pd.to_numeric(data_frame['Amount of strong beer etc. usually drunk on any one day (large cans) (Capi)'],
                          errors='coerce')
        data_frame['Amount of strong beer etc. usually drunk on any one day (bottles) (Capi)'].replace('Not applicable',
                                                                                                       0, inplace=True)
        data_frame['Amount of strong beer etc. usually drunk on any one day (bottles) (Capi)'] = \
            pd.to_numeric(data_frame['Amount of strong beer etc. usually drunk on any one day (bottles) (Capi)'],
                          errors='coerce')

        data_frame['Amount of sherry usually drunk on any one day (small glasses) (Capi)'].replace('Not applicable', 0,
                                                                                                   inplace=True)
        data_frame['Amount of sherry usually drunk on any one day (small glasses) (Capi)'].replace("Don't know", 0,
                                                                                                   inplace=True)
        data_frame['Amount of sherry usually drunk on any one day (small glasses) (Capi)'] = \
            pd.to_numeric(data_frame['Amount of sherry usually drunk on any one day (small glasses) (Capi)'],
                          errors='coerce')

        data_frame['Amount of spirits usually drunk on any one day (single measures) (Capi)'].replace('Refused',
                                                                                                      0, inplace=True)
        data_frame['Amount of spirits usually drunk on any one day (single measures) (Capi)'].replace('Not applicable',
                                                                                                      0, inplace=True)
        data_frame['Amount of spirits usually drunk on any one day (single measures) (Capi)'].replace("Don't know", 0,
                                                                                                      inplace=True)
        data_frame['Amount of spirits usually drunk on any one day (single measures) (Capi)'] = \
            pd.to_numeric(data_frame['Amount of spirits usually drunk on any one day (single measures) (Capi)'],
                          errors='coerce')
        data_frame['Heaviest day normal beer: Pints (Capi)'].replace('Not applicable', 'Not mentioned', inplace=True)
        data_frame['Heaviest day normal beer: Small cans (Capi)'].replace('Not applicable', 'Not mentioned',
                                                                          inplace=True)
        data_frame['Heaviest day normal beer: Large cans (Capi)'].replace('Not applicable', 'Not mentioned',
                                                                          inplace=True)
        data_frame['Heaviest day normal beer: Bottles (Capi)'].replace('Not applicable', 'Not mentioned', inplace=True)
        data_frame['Whether had drink in last 7 days (capi+casi)'].replace('Not applicable', 'No', inplace=True)
        data_frame['Drink now compared to 5 years ago (Capi)'].replace('Not applicable', 'Not a drinker', inplace=True)
        data_frame['12 months normal beer: Pints (Capi)'].replace('Not applicable', 'Not mentioned', inplace=True)
        data_frame['12 months normal beer: Small cans (Capi)'].replace('Not applicable', 'Not mentioned', inplace=True)
        data_frame['12 months normal beer: Large cans (Capi)'].replace('Not applicable', 'Not mentioned', inplace=True)
        data_frame['12 months normal beer: Bottles (Capi)'].replace('Not applicable', 'Not mentioned', inplace=True)
        data_frame['Freq of drinking normal beer etc. over last 12 months (Capi)'] \
            .replace('Not applicable', 'Not at all in the last 12 months', inplace=True)
        data_frame['Freq of drinking spirits over last 12 months (Capi)'] \
            .replace('Not applicable', 'Not at all in the last 12 months', inplace=True)
        data_frame['Freq of drinking strong beer etc. over last 12 months (Capi)'] \
            .replace('Not applicable', 'Not at all in the last 12 months', inplace=True)
        data_frame['Freq of drinking sherry over last 12 months (Capi)'] \
            .replace('Not applicable', 'Not at all in the last 12 months', inplace=True)
        data_frame['12 months strong beer: Pints (Capi)'].replace('Not applicable', 'Not mentioned', inplace=True)
        data_frame['12 months strong beer: Small cans (Capi)'].replace('Not applicable', 'Not mentioned', inplace=True)
        data_frame['12 months strong beer: Large cans (Capi)'].replace('Not applicable', 'Not mentioned', inplace=True)
        data_frame['12 months strong beer: Bottles (Capi)'].replace('Not applicable', 'Not mentioned', inplace=True)

        data_frame['Frequency drank any alcoholic drink last 12 mths (capi+casi)'].replace('Not applicable',
                                                                                           'Not at all in the last 12 months',
                                                                                           inplace=True)
        data_frame['Stopped drinking because of a particular health condition (Capi)'].replace('Not applicable',
                                                                                               'Not a drinker',
                                                                                               inplace=True)
        data_frame['Whether always non-drinker (capi+casi)'].replace('Not applicable', 'still a drinker', inplace=True)
        data_frame['Whether drinks occasionally or never drinks (capi+casi)'].replace('Not applicable', 'often',
                                                                                      inplace=True)

        data_frame['(D) Units of alcopops on heaviest day'].replace('Not applicable', 0, inplace=True)
        data_frame['(D) Units of alcopops on heaviest day'].replace("Don't know", 0, inplace=True)
        data_frame['(D) Units of alcopops on heaviest day'] = pd.to_numeric(
            data_frame['(D) Units of alcopops on heaviest day'], errors='coerce')
        data_frame['(D) Units of sherry on heaviest day'].replace('Not applicable', 0, inplace=True)
        data_frame['(D) Units of sherry on heaviest day'].replace("Don't know", 0, inplace=True)
        data_frame['(D) Units of sherry on heaviest day'] = pd.to_numeric(
            data_frame['(D) Units of sherry on heaviest day'], errors='coerce')
        data_frame['(D) Units of wine on heaviest day'].replace('Not applicable', 0, inplace=True)
        data_frame['(D) Units of wine on heaviest day'].replace("Don't know", 0, inplace=True)
        data_frame['(D) Units of wine on heaviest day'] = pd.to_numeric(data_frame['(D) Units of wine on heaviest day'],
                                                                        errors='coerce')
        data_frame['(D) Units of spirits on heaviest day'].replace('Not applicable', 0, inplace=True)
        data_frame['(D) Units of spirits on heaviest day'].replace("Don't know", 0, inplace=True)
        data_frame['(D) Units of spirits on heaviest day'] = pd.to_numeric(
            data_frame['(D) Units of spirits on heaviest day'], errors='coerce')
        data_frame['(D) Units of strong beer on heaviest day'].replace('Not applicable', 0, inplace=True)
        data_frame['(D) Units of strong beer on heaviest day'].replace("Don't know", 0, inplace=True)
        data_frame['(D) Units of strong beer on heaviest day'] = pd.to_numeric(
            data_frame['(D) Units of strong beer on heaviest day'], errors='coerce')

        data_frame['(D) Units of normal beer on heaviest day'].replace('Not applicable', 0, inplace=True)
        data_frame['(D) Units of normal beer on heaviest day'].replace("Don't know", 0, inplace=True)
        data_frame['(D) Units of normal beer on heaviest day'] = pd.to_numeric(
            data_frame['(D) Units of normal beer on heaviest day'], errors='coerce')

        column_numbers = DataController.get_dup_indices \
            (data_frame, '(D) Units drunk on heaviest day in last 7 (16yrs+)')
        data_frame = data_frame.iloc[:, column_numbers]
        column_numbers = DataController.get_dup_indices \
            (data_frame,
             "How often in the past six months had time off work because of problems with child's teeth, mouth or gums")
        data_frame = data_frame.iloc[:, column_numbers]
        data_frame.drop(columns=[
            "How often in the past six months had time off work because of problems with child's teeth, mouth or gums"],
            inplace=True)

        data_frame['(D) Units drunk on heaviest day in last 7 (16yrs+)'].replace('Not applicable', 'None', inplace=True)
        data_frame['(D) Units drunk on heaviest day in last 7 (16yrs+)'].replace("Don't know", 'None', inplace=True)

        # combine smokers columns and drop
        data_frame['(D) Number of people who smoke inside this house/flat on most days, top coded 6+'].replace(
            'Not applicable', 'No smokers', inplace=True)
        data_frame.drop(
            index=data_frame[
                data_frame["(D) Number of cigarettes smoke a day - inc. non-smokers"] == "Don't know"].index,
            inplace=True)
        data_frame.drop(
            index=data_frame[
                data_frame["(D) Number of cigarettes smoke a day - inc. non-smokers"] == "Not applicable"].index,
            inplace=True)
        data_frame.drop(index=data_frame[
            data_frame["Number of hours/week exposed to others' smoke (capi+casi)"] == "Don't know"].index,
                        inplace=True)
        data_frame.drop(index=data_frame[
            data_frame["Number of hours/week exposed to others' smoke (capi+casi)"] == "Not applicable"].index,
                        inplace=True)
        data_frame['Number of cigarettes smoke on weekday (capi+casi)'].replace('Not applicable', 0, inplace=True)
        data_frame['Number of cigarettes smoke on weekday (capi+casi)'] = pd.to_numeric(
            data_frame['Number of cigarettes smoke on weekday (capi+casi)'], errors='coerce')
        data_frame['Number of cigarettes hand rolled on weekday (capi+casi)'].replace('Not applicable', 0, inplace=True)
        data_frame['Number of cigarettes hand rolled on weekday (capi+casi)'] = pd.to_numeric(
            data_frame['Number of cigarettes hand rolled on weekday (capi+casi)'], errors='coerce')
        data_frame['Number of cigarettes smoke on weekend day (capi+casi)'].replace('Not applicable', 0, inplace=True)
        data_frame['Number of cigarettes smoke on weekend day (capi+casi)'] = pd.to_numeric(
            data_frame['Number of cigarettes smoke on weekend day (capi+casi)'], errors='coerce')
        data_frame['Number of cigarettes hand rolled on weekend day (capi+casi)'].replace('Not applicable', 0,
                                                                                          inplace=True)
        data_frame['Number of cigarettes hand rolled on weekend day (capi+casi)'].replace("Don't know", 0, inplace=True)
        data_frame['Number of cigarettes hand rolled on weekend day (capi+casi)'] = pd.to_numeric(
            data_frame['Number of cigarettes hand rolled on weekend day (capi+casi)'], errors='coerce')

        data_frame.drop(index=data_frame[
            data_frame[
                "(D) Any prescribed Antipsychotic medications taken in last 7 days (binary)"] == "Don't know"].index,
                        inplace=True)
        data_frame.drop(index=data_frame[
            data_frame[
                "(D) Any prescribed Antipsychotic medications taken in last 7 days (binary)"] == 'Not applicable'].index,
                        inplace=True)

        data_frame["Whether smoke cigarettes nowadays (capi+casi)"].replace('Not applicable', 'No', inplace=True)
        data_frame["Smoked any cigarettes in the last month (Capi)"].replace('Not applicable', 'No', inplace=True)
        data_frame["How often smoked a cigarette during the last month (Capi)"].replace('Not applicable', 'Not smoker',
                                                                                        inplace=True)
        data_frame["Type of cigarette smoke (capi+casi)"].replace('Not applicable', 'Not cigarette smoker',
                                                                  inplace=True)

        data_frame["Smoked in the last 7 days: At home, indoors (Capi)"].replace('Not applicable',
                                                                                 'Not smoked in last 7 days',
                                                                                 inplace=True)
        data_frame["Smoked in the last 7 days: At home, outside e.g. in garden or on doorstep (Capi)"].replace(
            'Not applicable', 'Not smoked in last 7 days', inplace=True)
        data_frame["Smoked in the last 7 days: Outside in the street or out and about (Capi)"].replace('Not applicable',
                                                                                                       'Not smoked in last 7 days',
                                                                                                       inplace=True)
        data_frame["Smoked in the last 7 days: Outside at work (Capi)"].replace('Not applicable',
                                                                                'Not smoked in last 7 days',
                                                                                inplace=True)

        data_frame["Smoked in the last 7 days: Outside at other people's homes (Capi)"].replace('Not applicable',
                                                                                                'Not smoked in last 7 days',
                                                                                                inplace=True)
        data_frame["Smoked in the last 7 days: Outside pubs, bars, restaurants or shops (Capi)"].replace(
            'Not applicable', 'Not smoked in last 7 days', inplace=True)
        data_frame["Smoked in the last 7 days: In public parks (Capi)"].replace('Not applicable',
                                                                                'Not smoked in last 7 days',
                                                                                inplace=True)
        data_frame["Smoked in the last 7 days: Inside other people's homes (Capi)"].replace('Not applicable',
                                                                                            'Not smoked in last 7 days',
                                                                                            inplace=True)

        data_frame["Smoked in the last 7 days: While travelling by car (Capi)"].replace('Not applicable',
                                                                                        'Not smoked in last 7 days',
                                                                                        inplace=True)
        data_frame["Smoked in the last 7 days: Inside other places (Capi)"].replace('Not applicable',
                                                                                    'Not smoked in last 7 days',
                                                                                    inplace=True)
        data_frame["How soon after waking does respondent smoke (Capi)"].replace('Not applicable', 'Not smoker',
                                                                                 inplace=True)
        data_frame["Number of cigarettes smoked compared to a year ago (capi+casi)"].replace('Not applicable',
                                                                                             'Not smoked cigarettes',
                                                                                             inplace=True)
        data_frame["Ease of going without cigarettes for a day (Capi)"].replace('Not applicable', 'Not smoker',
                                                                                inplace=True)

        data_frame["Like to give up smoking (capi+casi)"].replace('Not applicable', 'Not smoker', inplace=True)
        data_frame["Intention to stop smoking (capi+casi)"].replace('Not applicable', 'Not smoker', inplace=True)
        data_frame["Ever made a serious attempt to stop smoking completely (capi+casi)"].replace('Not applicable',
                                                                                                 'Not smoker',
                                                                                                 inplace=True)
        data_frame["Number of attempts made to stop smoking in the last 12 months (capi+casi)"].replace(
            'Not applicable', 0, inplace=True)
        data_frame["Main reasons for wanting to give up smoking: Better for health (capi+casi)"].replace(
            'Not applicable', 'Not smoker', inplace=True)
        data_frame[
            "Main reasons for wanting to give up smoking: Financial reasons/can't afford it (capi+casi)"].replace(
            'Not applicable', 'Not smoker', inplace=True)

        data_frame["Main reasons for wanting to give up smoking: Family/friends want me to stop (capi+casi)"].replace(
            'Not applicable', 'Not smoker', inplace=True)
        data_frame[
            "Main reasons for wanting to give up smoking: Worried about the effect on other people (capi+casi)"].replace(
            'Not applicable', 'Not smoker', inplace=True)
        data_frame[
            "Main reasons for wanting to give up smoking: Something else (capi+casi)"].replace(
            'Not applicable', 'Not smoker', inplace=True)
        data_frame[
            "Whether ever smoked cigarettes (capi+casi)"].replace(
            'Not applicable', 'No', inplace=True)
        data_frame[
            "Smoked cigarettes in the last month (CASI)"].replace(
            'Not applicable', 'No', inplace=True)
        data_frame[
            "Decided to give up smoking: For health reasons (capi+casi)"].replace(
            'Not applicable', 'Not smoker', inplace=True)
        data_frame[
            "Decided to give up smoking: Pregnancy (capi+casi)"].replace(
            'Not applicable', 'Not smoker', inplace=True)
        data_frame[
            "Decided to give up smoking: Financial reasons/couldn't afford it (capi+casi)"].replace(
            'Not applicable', 'Not smoker', inplace=True)

        data_frame[
            "Decided to give up smoking: Family/friends wanted me to stop (capi+casi)"].replace(
            'Not applicable', 'Not smoker', inplace=True)

        data_frame[
            "Decided to give up smoking: Worried about effect on other people (capi+casi)"].replace(
            'Not applicable', 'Not smoker', inplace=True)
        data_frame[
            "Decided to give up smoking: Own motivation (capi+casi)"].replace(
            'Not applicable', 'Not smoker', inplace=True)
        data_frame[
            "Decided to give up smoking: Something else (capi+casi)"].replace(
            'Not applicable', 'Not smoker', inplace=True)

        data_frame[
            "Decided to give up smoking: Cannot remember (capi+casi)"].replace(
            'Not applicable', 'Not smoker', inplace=True)
        data_frame[
            "Decided to give up smoking: Haven't stopped smoking (capi+casi)"].replace(
            'Not applicable', 'Not smoker', inplace=True)
        data_frame[
            "How frequently used to smoke (capi+casi)"].replace(
            'Not applicable', 'Not smoker', inplace=True)
        data_frame[
            "Age started smoking cigarettes regularly (Capi)"].replace(
            'Not applicable', 'Not smoker', inplace=True)
        data_frame[
            "Age first started to use e-cigarettes or vaping devices (capi+casi)"].replace(
            'Not applicable', 'Never used e-cigarettes/ vaping', inplace=True)
        data_frame[
            "Use e-cigarette or vaping device nowadays (capi+casi)"].replace(
            'Not applicable', 'No', inplace=True)

        list_of_blood_ec = ['How often used e-cigarette or vaping device in last month (capi+casi)',
                            'How soon after waking usually have first e-cigarette or vape of the day (capi+casi)',
                            'How many times use e-cigarette or vaping device on typical weekday (capi+casi)',
                            'Total time spend using e-cigarette or vaping device on typical weekday (capi+casi)',
                            'How many times use e-cigarette or vaping device on typical Saturday or Sunday (capi+casi)',
                            'Total time spend using e-cigarette or vaping device on typical Saturday or Sunday (capi+casi)',
                            'Type of e-cigarette or vaping device mainly use (capi+casi)',
                            'Strength of e-cigarette cartridge typically use (capi+casi)',
                            'Would like to give up using e-cigarettes or vaping altogether (capi+casi)',
                            'Used e-cigarette or vaping device in the last 7 days ending yesterday: At my home, indoors (Capi)',
                            'Used e-cigarette or vaping device in the last 7 days ending yesterday: At my home, outside e.g. in garden or on doorstep (Capi)',
                            'Used e-cigarette or vaping device in the last 7 days ending yesterday: Outside in the street or out and about (Capi)',
                            'Used e-cigarette or vaping device in the last 7 days ending yesterday: Outside at work (Capi)',
                            "Used e-cigarette or vaping device in the last 7 days ending yesterday: Outside at other people's homes (Capi)",
                            'Used e-cigarette or vaping device in the last 7 days ending yesterday: Outside pubs, bars, restaurants or shops (Capi)',
                            'Used e-cigarette or vaping device in the last 7 days ending yesterday: In public parks (Capi)',
                            "Used e-cigarette or vaping device in the last 7 days ending yesterday: Inside other people's homes (Capi)",
                            'Used e-cigarette or vaping device in the last 7 days ending yesterday: While travelling by car (Capi)',
                            'Used e-cigarette or vaping device in the last 7 days ending yesterday: Inside other places (Capi)',
                            'Started smoking cigarettes before or after started using e-cigarettes or vaping devices (Capi)',
                            'Compared to before started using e-cigarettes or vaping devices, currently smoke fewer, the same number, or more cigarettes (Capi)']
        DataController.replace_based_on_condition(data_frame, list_of_blood_ec,
                                                  'Use e-cigarette or vaping device nowadays (capi+casi)', 'No',
                                                  'Do not smoke e-cigarette')

        data_frame[
            "Used hookah or shisha in last month (capi+casi)"].replace(
            'Not applicable', 'No', inplace=True)
        data_frame[
            "Used non-smoked tobacco that you put in your mouth in the last month (capi+casi)"].replace(
            'Not applicable', 'No', inplace=True)
        data_frame[
            "(D) How long ago stopped smoking cigarettes (grouped years) (capi+casi)"].replace(
            'Not applicable', 'Not smoker', inplace=True)
        data_frame[
            "(D) Number of years smoked for (grouped) (capi+casi)"].replace(
            'Not applicable', 'Not smoker', inplace=True)

        # move target variable to last column.
        df1 = data_frame.pop(
            '(D) Any prescribed Antipsychotic medications taken in last 7 days (binary)')  # remove column b and store it in df1
        data_frame[
            '(D) Any prescribed Antipsychotic medications taken in last 7 days (binary)'] = df1  # add b series as a 'new' column.

        # change no into - and yes to 1
        data_frame[
            '(D) Any prescribed Antipsychotic medications taken in last 7 days (binary)'].replace('No', 0,
                                                                                                  inplace=True)
        data_frame[
            '(D) Any prescribed Antipsychotic medications taken in last 7 days (binary)'].replace('Yes, at least '
                                                                                                  'one', 1,
                                                                                                  inplace=True)
        data_frame['(D) Any prescribed Antipsychotic medications taken in last 7 days (binary)'] = pd.to_numeric(
            data_frame['(D) Any prescribed Antipsychotic medications taken in last 7 days (binary)'], errors='coerce')

        features = data_frame.columns
        for c in features:
            try:
                data_frame[c] = pd.to_numeric(data_frame[c])
            except:
                pass

        return data_frame

    # helper methods
    @staticmethod
    def replace_based_on_condition(data_frame, alist, target, target_value, new_value):
        for item in alist:
            data_frame[item] = np.where(data_frame[target] == target_value, new_value, data_frame[item])

    @staticmethod
    def convert_by_median(data_frame, target):
        data_frame[target].replace('Not applicable', np.NaN, inplace=True)
        data_frame[target] = pd.to_numeric(data_frame[target], errors='coerce')
        data_frame[target] = data_frame[target].fillna((data_frame[target].median()))

    @staticmethod
    def get_dup_indices(data_frame, target):
        indices = [i for i, x in enumerate(data_frame.columns) if
                   x == target]
        column_numbers = [x for x in range(data_frame.shape[1])]
        for i in range(len(indices) - 1):
            column_numbers.remove(indices[i])
        return column_numbers

    @staticmethod
    def write_dataframe_to_csv(file_name, dataset, index):
        if not index:
            dataset.to_csv(file_name + ".csv", encoding='utf-8')
        else:
            dataset.to_csv(file_name + ".csv", index=False, encoding='utf-8')

    @staticmethod
    def split_mean(x):
        split_list = x.split('-')
        mean = (float(split_list[0]) + float(split_list[1])) / 2
        return mean

    @staticmethod
    def open_file():
        """Open a file for editing."""
        filepath = askopenfilename(
            filetypes=[("All Files", "*.*"), ("Text Files", "*.csv")]
        )
        if not filepath:
            return
        else:
            return filepath

    @staticmethod
    def set_column_labels(data_frame):
        label_list = []
        try:
            with open("hse_2019_eul_20211006_ukda_data_dictionary.txt") as inFile:
                content = inFile.readlines()
                for line in content:
                    if line.find("Variable label = ") != -1:
                        indexline = line.rindex(" = ")
                        label = line[indexline + 3:]
                        label_list.append(label.strip('\n'))
            # problems with labels -4 and -6
            label_list.remove('Parents ever smoked regularly when a child (Capi)')  # 4
            label_list.remove('Parents smoked regularly as a child (CASI)')  # 4
            label_list.remove('Waist Hip Outcome')  # 6

            # use readable version of labels.
            data_frame.columns = label_list

            return data_frame

        except TypeError as te:
            raise TypeError(f"wrong type. {te}")
        except ValueError as ve:
            raise ValueError(f"File not in correct format. {ve}")

    @staticmethod
    def columns_to_remove_by_file(data_frame):

        data_frame = pd.DataFrame(data=data_frame)
        list_of_columns = []
        try:
            with open("list_of_columns.txt", newline='') as inFile:
                for line in inFile:
                    line = line.rstrip()
                    list_of_columns.append(line)

            for line in list_of_columns:
                data_frame.drop(columns=line, inplace=True)

            return data_frame

        except TypeError as te:
            raise TypeError(f"wrong type. {te}")
        except ValueError as ve:
            raise ValueError(f"File not in correct format. {ve}")

    @staticmethod
    def pickle_object(name, obj):
        with open(f"{name}.obj", "wb") as f:
            pickle.dump(obj, f)
        f.close()

    @staticmethod
    def load_pickled_object(name):
        if name is None:
            with open(DataController.open_file(), "rb") as f:
                obj = pickle.load(f)
            f.close()
        else:
            with open(name, "rb") as f:
                obj = pickle.load(f)
            f.close()
        return obj
