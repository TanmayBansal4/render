
# utilities\blob_utils.py

"""This modules implements blob functionalities"""
from azure.core.credentials import AzureSasCredential
from azure.storage.blob import BlobServiceClient
import os
import pandas as pd
import json
from ast import literal_eval

class BlobFunctionalities:

    def __init__(self) -> None:
        self._blob_service_client = BlobServiceClient(
                                                        account_url=os.getenv('AZURE_BLOB_ACCOUNT_URL'),
                                                        credential=AzureSasCredential(os.getenv('AZURE_BLOB_SAS_TOKEN'))
                                                    )
        self._container_client = self._blob_service_client.get_container_client(container=os.getenv('BLOB_CONTAINER_NAME'))
    
    def upload_file_in_blob_storage(self,file) -> bool:
        
        """Uploads file in blob storage
        """
        try:
            blob_client = self._container_client.upload_blob(file.filename,file.file.read())
            return True
        
        except Exception as e:
            print(e)
            return False
        
    def get_blob_url(self, blob_name: str) -> dict:
        """Retrieve from blob storage

        Args:
            blob_name (str): name of the blob to retrieve

        Returns:
            str: data link of the  file
        """
        if blob_name.endswith('.csv'):
            blob_client = self._container_client.get_blob_client(blob_name)
            return blob_client.url
    
    def read_csv_from_blob(self, blob_name: str) -> pd.DataFrame:
        """Read csv file from blob storage.

        Args:
            blob_name (str): name of the file.

        Returns:
            pd.DataFrame : dataframe with csv data
        """
        if blob_name.endswith('.csv'):
            blob_client = self._container_client.get_blob_client(blob_name)
            df = pd.read_csv(blob_client.download_blob())
            return df
    
    def read_file_from_blob(self, blob_name: str) -> list:
        """Read csv file from blob storage.

        Args:
            blob_name (str): name of the file.

        Returns:
            pd.DataFrame : dataframe with csv data
        """

        blob_client = self._container_client.get_blob_client(blob_name)
        data = blob_client.download_blob().content_as_text()
        # print(data)
        data = json.loads(data)
        return data
    
    def get_blob_client_exists(self,blob_name: str) -> bool:
        """This will provide the blob name suggested already exists or not in the blob storage account.

        Args:
            blob_name (str): name of the file on blob

        Returns:
            bool: True if exist otherwise False
        """
        return self._container_client.get_blob_client(blob=blob_name).exists()

    def df_to_csv_blob(self, df: pd.DataFrame, user_id: str,idx : str) -> str:
        """Convert Data frame to csv and store it on blob

        Args:
            blob_name (str): name of the file

        Returns:
            str: url to blob
        """
        file_data = df.to_csv()
        blob_client = self._container_client.upload_blob(name=str(user_id)+"_data_"+idx +".csv",data=file_data)
        return blob_client.url

    def upload_data_session(self,user_id : str, chat_session_id: str, data: dict) -> bool:
        """Uploads data sessions on Blob storage

        Args:
            user_id (str): User Id to generate name of the blob
            data (list): data be be updated on chat session data

        Returns:
            bool: successful or not
        """
        
        # Generate the name of BLOB
        user_name = str(user_id.split('@')[0].replace(".",""))
        blob_name = f'chat_history/{user_name}/active/{chat_session_id} +\'.json\''

        # Check for user is new or having chat sessions
        blob_exists = self.get_blob_client_exists(blob_name = blob_name)
        print("BLOB EXISTS:",blob_exists)

        # If New chat session
        if not blob_exists:

            session_id_blob = str(user_id.split('@')[0].replace(".","")) + '_chat_sessions.json'
            check_sessions = self.get_blob_client_exists(blob_name=session_id_blob)

            # If new user
            if not check_sessions:
                sessions = []
                sessions.append(
                    {
                        "chat_session_id" : chat_session_id,
                        "chat_session_name" : str(user_id.split('@')[0].replace(".","")) + '_1',
                        "user_id" : user_id
                    }
                )
                self._container_client.upload_blob(name= session_id_blob,data=json.dumps(sessions,default=str))
            
            # If existing user
            else:
                sessions = self.read_file_from_blob(blob_name=session_id_blob)
                sessions.append(
                        {
                            "chat_session_id" : chat_session_id,
                            "chat_session_name" : str(user_id.split('@')[0].replace(".","")) + '_' + str(len(sessions)+1),
                            "user_id" : user_id
                        }
                    )
                self._container_client.upload_blob(name= session_id_blob,data=json.dumps(sessions,default=str),overwrite =True)
                
            # Chat Session Data Upload
            prev_data:list = list()
            prev_data.append(data[0])
            prev_data.append(data[1])
            return self._container_client.upload_blob(name= blob_name,data=json.dumps(prev_data,default=str))
        
        else:

            # Retrieve Chat Session
            prev_data: list = self.read_file_from_blob(blob_name=blob_name)

            # Append data to chat session
            prev_data.append(data[0])
            prev_data.append(data[1])

            return self._container_client.upload_blob(name= blob_name,data=json.dumps(prev_data,default=str), overwrite = True)
        
    def get_chat_session(self,user_id : str, chat_session_id: str) -> dict:
        """Retrieves user chat session

        Args:
            user_id (str): user id of the user
            chat_session_id (str): chat session id from front end

        Returns:
            dict: chat session data
        """
        blob_name = 'chat_session_'+str(user_id.split('@')[0].replace(".",""))+'_' + chat_session_id +'.json'

        if self.get_blob_client_exists(blob_name=blob_name):
            chat_session_data = self.read_file_from_blob(blob_name=blob_name)
            return chat_session_data
        
        else:
            return {
                "Error": "No Chat Session Found"
            }

    def get_user_sessions(self, user_id: str) -> list:
        """Provides list of sessions based on user_id

        Args:
            user_id (str): user id of the user accessing Andon Copilot

        Returns:
            list: list of sessions user is having.
        """
        session_id_blob = str(user_id.split('@')[0].replace(".","")) + '_chat_sessions.json'
        check_sessions = self.get_blob_client_exists(blob_name=session_id_blob)
        if not check_sessions:
            return []
        
        else:
            return self.read_file_from_blob(session_id_blob)
        
    def check_for_new_user(self,user_id: str) -> bool:

        blob_data = self._container_client.download_blob('user_list/lla_user_list.json').content_as_text()
        user_data = json.loads(blob_data)
        if user_id in user_data['user_list']:
            return False
        else:
            "New user"
            user_data['user_list'].append(user_id)
            self._container_client.upload_blob(name='user_list/lla_list.json',data=json.dumps(user_data),overwrite=True)
            return True