#from hive_query import HiveQuery
import hive_query
from utils.config import get_config
import pandas as pd
import subprocess
import os
from time import time as t
import sys



# os.environ["AZCOPY_JOB_PLAN_LOCATION"]="/tf"
# os.environ["AZCOPY_LOG_LOCATION"]="/tf"

# config = get_config("production_config")
# connection = HiveQuery()
# curs = connection.conn.cursor()
# run_date = sys.argv[1]

# def select_week_data(connection, week):
#     sql_query = "SELECT * FROM latam_imageprocess.latam_img_pir_final where weeknumber='"+week+"'"
#     res = pd.read_sql(sql_query, connection.conn)
#     new_columns = {col: col.split("latam_img_pir_final.")[-1] for col in res.columns.values}
#     res.rename(columns=new_columns, inplace=True)
#     #display(res)
#     return res


# tab_data = select_week_data(connection,run_date)
# tab_data['before_image_name']=tab_data['before_image_uri'].str.split(pat="/")
# l = []
# for i in range(0,len(tab_data.index)):
#     l.append(".".join((tab_data.iloc[i]['before_image_name'][-1]).split(".")[:-1]))
# tab_data['before_image'] = l


# csv_name = "detections.csv"
# csv_location = os.path.join(config.get('path').get('blob_base_dir'), config.get('compliance').get('path').get('output_prev_after_comparsion'),csv_name)

# detections = pd.read_csv(csv_location)
# detections['detections'] = detections['Detections']
# detections['before_image'] = detections['Image Name']

# res = pd.merge(tab_data,detections,on='before_image',how='left')
# res['detections'] = res['detections_y']
# res['detections'] = res['detections'].apply(lambda x : str(x).replace(",", ";"))
# res.drop(['Detections', 'Image Name', 'detections_x','detections_y','before_image_name','before_image'],axis=1,inplace=True)
# columnsTitles = ['routeid' , 'storeid' , 'salesrepid' , 'gpscoords' , 'rack' , 'img_date' , 'before_image_uri' , 'after_image_uri' , 'detections' , 'weeknumber']
# res = res.reindex(columns=columnsTitles)
# model_op = os.path.join(config.get("path").get("data"),"model_outputs.csv")
# res.to_csv(model_op, index=False, header=False)

# def upload_table_to_blob(filepath=model_op):
#     p = subprocess.check_output([config.get('path').get('blob_base_dir') + "/./azcopy", 
#                     "copy", 
#                     filepath,
#                     "https://latamrmprdblob.blob.core.windows.net/suggestedorder/latam_img_tnx_stg/?sp=racwl&st=2020-07-21T21:42:23Z&se=2038-07-22T21:42:00Z&sv=2019-12-12&sr=c&sig=rLtVSg1pzyrb4BbOvBMRZ6lWqMmvqK03vTF%2FAKz2bUE%3D", 
#                     "--recursive"])


# def update_datalake_table(curs): 
#     t_init = t()
#     #load the csv into the staging table
#     print("processing the staging table...")
#     sql = "TRUNCATE TABLE latam_imageprocess.latam_img_pir_stg"
#     curs.execute(sql)
#     sql = "load data inpath 'wasbs://suggestedorder@latamrmprdblob.blob.core.windows.net/latam_img_tnx_stg/model_outputs.csv' into table latam_imageprocess.latam_img_pir_stg"
#     curs.execute(sql)

#     #Populate the final table with new results
#     print("saving the data in the final table...")
#     sql = "delete from latam_imageprocess.latam_img_pir_final WHERE weeknumber='"+run_date+"'"
#     curs.execute(sql)
#     sql = "INSERT into TABLE latam_imageprocess.latam_img_pir_final PARTITION(weeknumber='"+run_date+"""') SELECT routeid, storeid, salesrepid, gpscoords, rack, img_date, before_image_uri, after_image_uri, regexp_replace(detections,";",",") FROM latam_imageprocess.latam_img_pir_stg"""
#     curs.execute(sql)

#     print( "time consumed loading the data into staging and after into final table %.2f" % (t()-t_init) )


# upload_table_to_blob(model_op)
# update_datalake_table(curs)
