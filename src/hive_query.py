import pandas as pd
import jaydebeapi


# class HiveQuery:
#     def __init__(self,
#                  url = 'jdbc:hive2://prdhadooplatamrm.azurehdinsight.net:443/default;transportMode=http;ssl=true;httpPath=/hive2',
#                  user = 'Dev_team',
#                  pwd = 'Pepsico1234!',
#                  jar_route = "/mnt/common/hive-jdbc-1.2.1-standalone.jar"
#                  ):
#         """
#          Parameters
#         ----------
#         url : TYPE, string
#             The URL for the Data Lake, such as dev, QA o prod.
#         user : TYPE, string
#             The user to access the Data Lake.
#         pwd : TYPE, string
#             The password to access the Data Lake. 
#         jar_route : TYPE, string
#             The route where the hive-jdbc-1.2.1-standalone.jar file is located. 
#         Returns
#         -------
#         None.
#         """
#         self.conn = jaydebeapi.connect('org.apache.hive.jdbc.HiveDriver', 
#                                        url,
#                                        [user, pwd],
#                                        jar_route)

#     def make_query(self, sql_query):
#         """
#         This function performs the query given in the parameters, 
#         returns the result and save it into an internal variable named last_result
#         Parameters
#         ----------
#         sql_query : string
#             The query for the Data Lake connection.
#         Returns
#         -------
#             the result in pandas dataframe format
#         """
#         self.last_result = pd.read_sql(sql_query, self.conn)
#         return self.last_result
    
#     def save_last_result(self, name_path = "resultado.csv"):
#         """
#         This function saves the last result into a pandas dataframe
        
#         Parameters
#         ----------
#             name_path: The full path for the file to be saved
            
#         Returns
#         -------
#         None.
#         """
#         self.last_result.to_csv(name_path)
        
#     def tables(self, schema="DWL_P_INTL"):
#         """
#         This function returns the tables found for a given schema
        
#         Parameters
#         ----------
#             schema: The name of the schema to look for
        
#         Returns
#         -------
#             A pandas dataframe containing the list of the tables for the given schema
#         """
        
#         sql_query = """
#             SHOW TABLES
#             FROM {}
#             """.format(schema)
        
#         return pd.read_sql(sql_query, self.conn)

