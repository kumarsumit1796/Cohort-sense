from flask import Flask
from flask_restplus import Api, Resource
from flask_cors import CORS
from flask import send_file
from smart_open import smart_open
from csv import reader
import json
import pandas as pd
import time
import ast
import shutil
import warnings
import os
from itertools import chain
import math
import boto3
from boto3 import Session
import sys
sys.path.append("apis")
from feature_scaling import FeatureScaling
from feature_selection import FeatureSelection
from encode import DataEncode
from model_selection import ModelSelection
from adanet_model import AdanetModelBuilding
from cohort_sense_ml import Data_Fetching


currentDirPath = os.getcwd()
warnings.filterwarnings("ignore")

application = Flask(__name__)
CORS(application)
API_NAME = Api(application, version="1.0",
               title="COHORT SENSE",
               default="/",
               default_label="cohortanalyzer",
               description="Swagger Documentation",
               strict_slashes=False)
application.config['SECRET_KEY'] = 'disable the web security'
application.config['CORS_HEADERS'] = 'Content-Type'


class DataPreparation(Resource):
    @staticmethod
    def get(argsdf):
        args_list = argsdf.split(",")
        user_id = args_list[0]
        if len(args_list) == 3:
            disease_name = [args_list[1]]
            filter_val = [args_list[2]]
        elif len(args_list) == 5:
            disease_name = [args_list[1],args_list[2]]
            filter_val = [args_list[3],args_list[4]]
        else:
            sys.exit("Please Select almost 2 diseases")

        print("User Dragged files name and Filter Values : ",disease_name,filter_val)


        disease_tables = {'Type2Diabetes': {'tables': ["cs_patient_t2d", "cs_t2d_patients_id", "cs_obs_yes_t2d","cs_med_yes_t2d","cs_obs_no_t2d","cs_med_no_t2d"]},
                   'ChronicKidneyDisease': {'tables': ["cs_patient_ckd", "cs_ckd_patients_id", "cs_obs_ckd_yes","cs_med_ckd_yes","cs_obs_no_ckd","cs_med_no_ckd"]}
                   }




        data = []
        tables_name = []
        start=time.time()

        if len(disease_name) == 1 and filter_val[0] !='0' and filter_val[0] != '1' :
            print("Fetching observations and medications data for "+disease_name[0]+"...")
            tables_name.append(disease_tables[disease_name[0]]['tables'])
            tables_name = list(chain(*tables_name))
            final_data = Data_Fetching.single_disease(tables_name)
            print("Data Fetched.")
        elif len(disease_name) == 2 and filter_val[0] == '1'and filter_val != 1:
            tables_name.append(disease_tables[disease_name[0]]['tables'][0:4])
            tables_name.append(disease_tables[disease_name[1]]['tables'][0:2])
            tables_name = list(chain(*tables_name))
            final_data = Data_Fetching.double_disease(tables_name)
        else:
            sys.exit("Print select atmost 2 Diseases and Select an appropriate filter option !",len(disease_name),filter_val)
            final_data = pd.DataFrame()

        print("Final data shape ------->   ",final_data.shape)
        print("Values count ------->   ",final_data.target.value_counts())
        print("Final data top 5 ----->   ",final_data.head())

        del final_data['person_id']

        final_data.to_csv(currentDirPath+"/datafile/user_input.csv",index=False)

        s3_resource = boto3.resource('s3')
        s3_resource.Object("bucketname", "CohortAnalyzer/"+"COHORT_SENSE_"+user_id+".csv").upload_file(
            Filename=currentDirPath+"/datafile/user_input.csv")


        print("******************** Final CSV file has successfully stored in S3 bucket ******************")


class PreviewDataModule(Resource):
    @staticmethod
    def get(argspd):
        args_list = argspd.split(",")
        req_filename = args_list[0]
        req_filename = req_filename+'.csv'

        client = boto3.client('s3') #low-level functional API

        resource = boto3.resource('s3') #high-level object-oriented API
        my_bucket = resource.Bucket('bucketname')
        obj = client.get_object(Bucket='bucketname', Key='CohortAnalyzer/' + req_filename)
        data = pd.read_csv(obj['Body'])

        print(data.head())

        print("Data Encoding...")
        data.columns = data.columns.str.replace(' ', '_')
        col_dtypes_list = list(data.select_dtypes(include=['object']).columns)
        for c_name_rd in col_dtypes_list:
            data[c_name_rd] = data[c_name_rd].str.replace(' ', '_')
        for c_name in col_dtypes_list:
            if data[c_name].nunique() == 2:
                data = DataEncode.label_encode(data, c_name)
            else:
                data = DataEncode.one_hot_encode(data, c_name)
        print("Encoding has done..")

        print(data.head())

        data_copy = data.drop(columns=['target'], axis=1)
        data_copy['target'] = data['target']
        ctime = str(int(time.time()))
        filename = args_list[0] + ctime + '.csv'
        file_path = currentDirPath + "/csvfiles/" + filename
        jfilename = args_list[0] + ctime + '.json'
        json_path = currentDirPath + "/jsonfiles/" + jfilename
        data_copy.to_csv(file_path, index=False)
        data_t10 = data.head(10)
        with open(json_path, 'w') as jcon:
            json.dump({'FileName':req_filename}, jcon)
        print("Total columns----> ", data_t10.columns.tolist())
        print("File_name---->",filename, jfilename)
        return {'predata': data_t10.to_json(orient='table'), 'filename': filename, 'jfilename': jfilename}


class FeatureScalingModule(Resource):
    @staticmethod
    def get(argsfscale):
        print("Feature Scaling...")
        args_list = argsfscale.split(",")
        fsn = args_list[0]
        filename = args_list[1]
        jfilename = args_list[2]
        json_path = currentDirPath + "/jsonfiles/" + jfilename
        file_path = currentDirPath + "/csvfiles/" + filename
        data = pd.read_csv(file_path)
        data_afs = FeatureScaling.Scaling(data, fsn)
        data_afs.reset_index(inplace=True, drop=True)
        data_afs.to_csv(file_path, index=False)
        data_t10 = data_afs.head(10)
        data_noff = len(data_afs.columns)
        with open(json_path) as jcon:
            jdata = json.load(jcon)
        jdata.update({'FeatureScaling':fsn})
        with open(json_path, 'w') as j_con:
            json.dump(jdata, j_con)
        return {'predata': data_t10.to_json(orient='table'), 'fileName': filename, 'numofcol': data_noff, 'jfilename':jfilename}


class FeatureSelectionModule(Resource):
    @staticmethod
    def get(argsfselection):
        args_list = argsfselection.split(",")
        if len(args_list) == 4:
            feasel_name = args_list[0]
            feasel_arg = args_list[1]
            filename = args_list[2]
            jfilename = args_list[3]
            json_path = currentDirPath + "/jsonfiles/" + jfilename
            file_path = currentDirPath + "/csvfiles/" + filename
            data = pd.read_csv(file_path)
            data_afs = FeatureSelection.selection(data, feasel_name, feasel_arg)
            data_afs.reset_index(inplace=True, drop=True)
            data_afs.to_csv(file_path, index=False)
            data_t10 = data_afs.head(10)
            data_noff = len(data_afs.columns)
            with open(json_path) as jcon:
                jdata = json.load(jcon)
            jdata.update({'FeatureSelection': feasel_name})
            with open(json_path, 'w') as j_con:
                json.dump(jdata, j_con)
            return {'predata': data_t10.to_json(orient='table'), 'filename': filename, 'numofcol': data_noff, 'jfilename':jfilename}
        elif len(args_list) == 3:
            feasel_name = args_list[0]
            filename = args_list[1]
            jfilename = args_list[2]
            json_path = currentDirPath + "/jsonfiles/" + jfilename
            file_path = currentDirPath + "/csvfiles/" + filename
            data = pd.read_csv(file_path)
            data_afs = FeatureSelection.selection(data, feasel_name, "")
            data_afs.reset_index(inplace=True, drop=True)
            data_afs.to_csv(file_path, index=False)
            data_t10 = data_afs.head(10)
            data_noff = len(data_afs.columns)
            with open(json_path) as jcon:
                jdata = json.load(jcon)
            jdata.update({'FeatureSelection': feasel_name})
            with open(json_path, 'w') as j_con:
                json.dump(jdata, j_con)
            return {'predata': data_t10.to_json(orient='table'), 'filename': filename, 'numofcol': data_noff, 'jfilename':jfilename}


class ADANETModule(Resource):
    @staticmethod
    def get(argsadanet):
        args_list = argsadanet.split(",")
        learning_rate = args_list[0]
        train_steps = args_list[1]
        batch_size = args_list[2]
        learn_mixture_weights = args_list[3]
        adanet_lambda = args_list[4]
        adanet_iterations = args_list[5]
        random_seed = args_list[6]
        tts = args_list[7]
        filename = args_list[8]
        jfilename = args_list[9]
        json_path = currentDirPath + "/jsonfiles/" + jfilename
        file_path = currentDirPath + "/csvfiles/" + filename
        data = pd.read_csv(file_path)
        model_dir = currentDirPath + "/adaNetModels"
        max_iteration_steps = int(train_steps) // int(adanet_iterations)
        train_test = ModelSelection.split_train_test(data, tts)
        shutil.rmtree(model_dir, ignore_errors=True)
        results, _ = AdanetModelBuilding.train_and_evaluate(2, float(learning_rate),
                                                            json.loads(learn_mixture_weights.lower()),
                                                            int(adanet_lambda), max_iteration_steps, int(train_steps),
                                                            int(batch_size), int(random_seed), train_test, model_dir)
        with open(json_path) as jcon:
            jdata = json.load(jcon)
        jdata.update({'ModelName': 'Adanet', 'Learning_Rate': learning_rate, 'Train_Steps': train_steps,
                      'Batch_Size': batch_size, 'Learn_Mixture_Weights': learn_mixture_weights,
                      'Adanet_Lambda': adanet_lambda, 'Adanet_Iterations': adanet_iterations,
                      'Random_Seed': random_seed, 'Train_Test_Split': tts})
        with open(json_path, 'w') as j_con:
            json.dump(jdata, j_con)
        return {'Accuracy': str(results['accuracy']), 'AUC': str(results['auc']), 'Loss': str(results['loss']),
                'Precision': str(results['precision']), 'Recall': str(results['recall']), 'filename': filename, 'jfilename':jfilename}


class AKModule(Resource):
    @staticmethod
    def get(argsautokeras):
        args_list = argsautokeras.split(",")
        tts = args_list[0]
        filename = args_list[1]
        jfilename = args_list[2]
        json_path = currentDirPath + "/jsonfiles/" + jfilename
        file_path = currentDirPath + "/csvfiles/" + filename
        data = pd.read_csv(file_path)
        train_test = ModelSelection.split_train_test(data, tts)
        path = currentDirPath + "/autoKerasModels"
        predict, accuracy = AutoKerasModule.predict_and_evaluate(train_test, path)
        acc, precision, recall = AutoKerasModule.eval_metrics(train_test[3], predict)
        with open(json_path) as jcon:
            jdata = json.load(jcon)
        jdata.update({'ModelName': 'AutoKeras', 'Train_Test_Split':tts})
        with open(json_path, 'w') as j_con:
            json.dump(jdata, j_con)
        return {'Accuracy': str(acc), 'Precision': str(precision), 'Recall': str(recall), 'filename':filename, 'jfilename':jfilename}


class TraditionalModule(Resource):
    @staticmethod
    def get(argstm):
        args_list = argstm.split(",")
        tts = args_list[0]
        filename = args_list[1]
        jfilename = args_list[2]
        json_path = currentDirPath + "/jsonfiles/" + jfilename
        file_path = currentDirPath + "/csvfiles/" + filename
        data = pd.read_csv(file_path)
        data_list = ModelSelection.split_train_test(data, tts)
        auc_metrics_name, auc_metrics, all_met, all_cm, modelpicname, rocgraphlist = ModelSelection.compare_models(data_list)
        with open(json_path) as jcon:
            jdata = json.load(jcon)
        jdata.update({'ModelName': 'TraditionalModels', 'Train_Test_Split':tts})
        with open(json_path, 'w') as j_con:
            json.dump(jdata, j_con)
        return {'ModelName': auc_metrics_name, 'AUC': auc_metrics,
                'allmetrics': all_met, 'ConfusionMetr': all_cm, 'filename': filename, 'picklefn': modelpicname,
                'jfilename':jfilename, 'rocgraphdata': rocgraphlist}


class ModelTuningModule(Resource):
    @staticmethod
    def get(argsmt):
        args_list = []
        inli = [argsmt]
        for args_list in reader(inli):
            print(args_list)
        datasize = args_list[0]
        modelname = args_list[1]
        auc_metrics = args_list[2]
        all_met = args_list[3]
        all_met = ast.literal_eval(all_met)
        auc_metrics_name = ["logistic_regression","random_forest","adaboost"]
        tts = args_list[4]
        filename = args_list[5]
        jfilename = args_list[6]
        json_path = currentDirPath + "/jsonfiles/" + jfilename
        file_path = currentDirPath + "/csvfiles/" + filename
        data = pd.read_csv(file_path)
        data_list = ModelSelection.split_train_test(data, tts)
        record_size = datasize
        if len(record_size) != 0:
            reduced_df = data.sample(n=int(record_size), frac=None, random_state=3)
            data_list1 = ModelSelection.split_train_test(reduced_df, tts)
        else:
            data_list1 = data_list
        print(
            "Select model name to fine tune their metrics \n 1. logistic_regression\n 2. random_forest\n 3. adaboost "
            "\n else it takes the model having highest AUC")
        mn_ui = modelname
        if len(mn_ui) != 0:
            mn, ma, rasm, mr, mp, mc, picmodelname = ModelSelection.run_tuning(mn_ui, data_list, data_list1, all_met)
            with open(json_path) as jcon:
                jdata = json.load(jcon)
            jdata.update({'TunningModelName': mn_ui, 'Train_Test_Split': tts})
            with open(json_path, 'w') as j_con:
                json.dump(jdata, j_con)
        else:
            max_auc_pos = auc_metrics.index(max(auc_metrics))
            max_auc_mn = auc_metrics_name[max_auc_pos]
            mn, ma, rasm, mr, mp, mc, picmodelname = ModelSelection.run_tuning(max_auc_mn, data_list, data_list1, all_met)
            with open(json_path) as jcon:
                jdata = json.load(jcon)
            jdata.update({'TunningModelName': max_auc_mn, 'Train_Test_Split': tts})
            with open(json_path, 'w') as j_con:
                json.dump(jdata, j_con)
        return {'ModelName': mn, 'ModelAccuracy': ma, 'ROCScore': rasm, 'ModelRecall': mr, 'ModelPrecision': mp,
                'ModelCM': mc.tolist(), 'filename': filename, 'picmodelname': picmodelname, 'jfilename':jfilename}



class ExportDataModule(Resource):
    @staticmethod
    def get(modelParams):
        # modelFile = modelParams
        path = "./pickeldir/" + modelParams
        if os.path.isfile(path):
            print("file avilable")
            return send_file(path, as_attachment=True)
        else:
            print("file not available")
            return {'message': 'Not found'}, 404


API_NAME.add_resource(DataPreparation,'/datapreparation/<argsdf>',methods=['POST','GET'])
API_NAME.add_resource(PreviewDataModule, '/cohortanalyzer/previewdata/<argspd>', methods=['POST', 'GET'])
API_NAME.add_resource(FeatureScalingModule, '/cohortanalyzer/featurescaling/<argsfscale>', methods=['POST', 'GET'])
API_NAME.add_resource(TraditionalModule, '/cohortanalyzer/tm/<argstm>', methods=['POST', 'GET'])
API_NAME.add_resource(AKModule, '/cohortanalyzer/autokeras/<argsautokeras>', methods=['POST', 'GET'])
API_NAME.add_resource(ADANETModule, '/cohortanalyzer/adanet/<argsadanet>', methods=['POST', 'GET'])
API_NAME.add_resource(ExportDataModule, '/cohortanalyzer/exportModel/<modelParams>', methods=['POST', 'GET'])
API_NAME.add_resource(FeatureSelectionModule,'/cohortanalyzer/featureselection/<argsfselection>', methods=['POST', 'GET'])
API_NAME.add_resource(ModelTuningModule, '/cohortanalyzer/mt/<argsmt>', methods=['POST', 'GET'])

if __name__ == '__main__':
    application.run(host='0.0.0.0',port=5001)
