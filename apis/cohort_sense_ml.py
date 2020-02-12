import psycopg2
import pandas as pd
from datetime import datetime
from encode import DataEncode
import math
import time
import warnings
warnings.filterwarnings("ignore")


class Data_Fetching:
    @staticmethod
    #user defined function to retrieve single disease data
    def single_disease(tables_name):
        # Create Engine for Redshift to get data from Database XXXXX
        engine = psycopg2.connect(dbname='XXXXX', host='XXXXXXX', port='XXXX',
                                  user='XXXX', password='XXXX')
        start=time.time()
        yes_obs = pd.read_sql("SELECT * FROM "+tables_name[2],con=engine)
        end=time.time()
        print("Yes observations : ",end-start)

        start=time.time()
        yes_med = pd.read_sql("SELECT * FROM "+tables_name[3],con=engine)
        end=time.time()
        print("Yes medications : ",end-start)

        start=time.time()
        no_obs = pd.read_sql("SELECT * FROM " + tables_name[4], con=engine)
        end=time.time()
        print("No obersvations : ",end-start)

        start=time.time()
        no_med = pd.read_sql("SELECT * FROM " + tables_name[5], con=engine)
        end=time.time()
        print("No medications : ",end-start)

        observations = yes_obs.append(no_obs, ignore_index=True)
        medications = yes_med.append(no_med, ignore_index=True)

        obs_pivot = Data_Fetching.pivot_obs(observations)
        med_pivot = Data_Fetching.pivot_med(medications)


        start=time.time()
        # Extracting Demographic details of all patients
        demo = pd.read_sql("""SELECT * FROM """ + tables_name[0], con=engine)
        end=time.time()
        print("Demographic Data : ", end-start)
        # Find current age from date_of_birth column and delete date_of_birth column
        demo['date_of_birth'] = pd.to_datetime(demo['date_of_birth'])
        now = datetime.now()
        demo['age'] = (now - demo['date_of_birth']).astype('<m8[Y]')
        demo = demo.drop(['date_of_birth'], axis=1)

        yes_id_list = yes_obs['person_id'].unique().tolist()

        demo_yes = demo[demo.person_id.isin(yes_id_list)]
        demo_yes.insert(1, 'target', 1)
        demo_no = demo[~demo.person_id.isin(yes_id_list)]
        demo_no.insert(1, 'target', 0)
        demo_with_target = demo_yes.append(demo_no, ignore_index=True)

        obs_demo = pd.merge(obs_pivot, demo_with_target, on='person_id', how='inner')
        obs_demo_med = pd.merge(obs_demo, med_pivot, on='person_id', how='inner')
        start=time.time()
        #final_data = Data_Fetching.data_preprocessing(obs_demo_med)
        final_data=obs_demo_med.fillna(0)

        end=time.time()
        print("Data preprocessing : ",end-start)
        print("***********************Data fetching job done for single disease*************************")
        return final_data

    @staticmethod
    def double_disease(tables_name):
        engine = psycopg2.connect(dbname='XXXX', host='XXXXXXXX', port='XXXX',
                                  user='XXXX', password='XXXXXX')
        disease2_id_df = pd.read_sql("SELECT DISTINCT person_id FROM "+tables_name[5],con=engine)
        disease2_id_list = disease2_id_df.person_id.unique().tolist()

        start=time.time()
        yes_obs = pd.read_sql("""SELECT * FROM """+tables_name[2],con=engine)
        end=time.time()
        print("Yes observations of first disease: ",end-start)

        start=time.time()
        yes_med = pd.read_sql("""SELECT * FROM """+tables_name[3],con=engine)
        end=time.time()
        print("Yes medications of first disease: ",end-start)

        obs_pivot = Data_Fetching.pivot_obs(yes_obs)
        med_pivot = Data_Fetching.pivot_med(yes_med)

        start=time.time()
        # Extracting Demographic details of given cohort patients
        demo = pd.read_sql("""SELECT * FROM """ + tables_name[0], con=engine)
        end=time.time()
        print("Demographic Data of Disease 1 : ", end-start)

        # Find current age from date_of_birth column and delete date_of_birth column
        demo['date_of_birth'] = pd.to_datetime(demo['date_of_birth'])
        now = datetime.now()
        demo['age'] = (now - demo['date_of_birth']).astype('<m8[Y]')
        demo = demo.drop(['date_of_birth'], axis=1)


        demo_yes = demo[demo.person_id.isin(disease2_id_list)]
        demo_yes.insert(1, 'target', 1)
        demo_no = demo[~demo.person_id.isin(disease2_id_list)]
        demo_no.insert(1, 'target', 0)
        demo_with_target = demo_yes.append(demo_no, ignore_index=True)

        obs_demo = pd.merge(obs_pivot, demo_with_target, on='person_id', how='inner')
        obs_demo_med = pd.merge(obs_demo, med_pivot, on='person_id', how='inner')

        start=time.time()
        #final_data = Data_Fetching.data_preprocessing(obs_demo_med)
        final_data=obs_demo_med.fillna(0)
        final_data = DataEncode.label_encode(final_data, "gender")
        final_data = DataEncode.one_hot_encode(final_data, "race")
        final_data = DataEncode.one_hot_encode(final_data, "socioeconomic_status")
        end=time.time()
        print("Data preprocessing: ",end-start)
        return final_data



    @staticmethod
    def pivot_obs(observations):
        observations = observations[observations['type'] == 'numeric']
        observations['value'] = observations['value'].astype('float')
        observations.rename(columns={'start': 'date'}, inplace=True)

        observations = observations.sort_values(['person_id', 'code', 'date'], ascending=False)
        observations.reset_index(inplace=True, drop=True)
        observations = observations.groupby(['person_id', 'code']).head(3)
        obs_temp_agg_mean = observations[['person_id', 'code', 'value']].groupby(['person_id', 'code']).agg({'value': "mean"}).reset_index()
        # Get the mean CV observations for the duration of the condition
        observations_agg_mean_pivot = obs_temp_agg_mean.pivot(index='person_id', columns='code',values='value').reset_index()
        return observations_agg_mean_pivot

    @staticmethod
    def pivot_med(medications):
        medications.insert(3, 'value', 1)

        medications_agg = medications[['person_id', 'code', 'value']].groupby(['person_id', 'code']).agg({'value': 'max'}).reset_index()
        # Creating features by pivot the table
        medications_pivot = medications_agg.pivot(index='person_id', columns='code', values='value').reset_index()
        # zero imputation
        medications_pivot = medications_pivot.fillna(0)
        return medications_pivot


    @staticmethod
    def data_preprocessing(final_data):
        final_data['race'] = final_data['race'].astype('str')
        final_data['gender'] = final_data['gender'].astype('str')
        # Label encoding to the categorical column gender
        le = LabelEncoder()
        final_data["gender"] = le.fit_transform(final_data["gender"])
        final_data["race"] = le.fit_transform(final_data["race"])
        cols = list(final_data.columns)
        remove = ['person_id', 'target', 'gender', 'age', 'race']
        CV_contFeatures = [x for x in cols if x not in remove]
        # print("Age range from ",min(final_data['age'].unique()),'to',max(final_data['age'].unique()))
        age_grp_ci = [-1, 29, 39, 49, 59, 69, 110]

        # Mean Imputation on CV
        def meanImputation(df, CVFeatures, age_grp):
            d = pd.DataFrame()
            for cv in CVFeatures:
                for i in range(len(age_grp) - 1):
                    for j in range(2):
                        value = df[(df['age'] > age_grp[i]) & (df['age'] <= age_grp[i + 1]) & (df['gender'] == j)][
                            cv].mean()
                        temp = df[(df['age'] > age_grp[i]) & (df['age'] <= age_grp[i + 1]) & (df['gender'] == j)]
                        if (math.isnan(value)):
                            temp[cv] = temp[cv].fillna(0)
                        else:
                            temp[cv] = temp[cv].fillna(value)
                        d = pd.concat([d, temp], axis=0)
                df = d.copy()
                d = pd.DataFrame()
            return df

        final_data = meanImputation(final_data, CV_contFeatures, age_grp_ci)
        final_data = final_data.sort_index()
        # final_data['person_id']
        # Moving predicting columns at the last
        df_new = final_data.pop('target')
        final_data['target'] = df_new
        return final_data
