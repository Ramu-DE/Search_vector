#!/usr/bin/env python3
"""
Setup OpenSearch Serverless Collection for Vector Search
Creates collection, configures access, and initializes index
"""

import boto3
import time
import json
from typing import Dict, Any, Optional
from config import Config


class OpenSearchSetup:
    def __init__(self):
        self.region = Config.AWS_REGION
        self.aoss_client = boto3.client('opensearchserverless', region_name=self.region)
        self.sts_client = boto3.client('sts', region_name=self.region)

        # Get current IAM identity
        identity = self.sts_client.get_caller_identity()
        self.account_id = identity['Account']
        self.caller_arn = identity['Arn']

        # Extract role ARN (works for assumed roles)
        if ':assumed-role/' in self.caller_arn:
            # Convert assumed-role ARN to role ARN
            role_name = self.caller_arn.split('/')[-2]
            self.principal_arn = f"arn:aws:iam::{self.account_id}:role/{role_name}"
        else:
            self.principal_arn = self.caller_arn

        print(f"Account ID: {self.account_id}")
        print(f"Principal ARN: {self.principal_arn}")

    def create_encryption_policy(self, collection_name: str) -> bool:
        """Create encryption policy for the collection"""
        policy_name = f"{collection_name}-enc"

        policy = {
            "Rules": [
                {
                    "ResourceType": "collection",
                    "Resource": [f"collection/{collection_name}"]
                }
            ],
            "AWSOwnedKey": True
        }

        try:
            self.aoss_client.create_security_policy(
                name=policy_name,
                type='encryption',
                policy=json.dumps(policy)
            )
            print(f"✓ Created encryption policy: {policy_name}")
            return True
        except self.aoss_client.exceptions.ConflictException:
            print(f"  Encryption policy already exists: {policy_name}")
            return True
        except Exception as e:
            print(f"✗ Error creating encryption policy: {e}")
            return False

    def create_network_policy(self, collection_name: str) -> bool:
        """Create network policy for the collection"""
        policy_name = f"{collection_name}-net"

        policy = [
            {
                "Rules": [
                    {
                        "ResourceType": "collection",
                        "Resource": [f"collection/{collection_name}"]
                    },
                    {
                        "ResourceType": "dashboard",
                        "Resource": [f"collection/{collection_name}"]
                    }
                ],
                "AllowFromPublic": True
            }
        ]

        try:
            self.aoss_client.create_security_policy(
                name=policy_name,
                type='network',
                policy=json.dumps(policy)
            )
            print(f"✓ Created network policy: {policy_name}")
            return True
        except self.aoss_client.exceptions.ConflictException:
            print(f"  Network policy already exists: {policy_name}")
            return True
        except Exception as e:
            print(f"✗ Error creating network policy: {e}")
            return False

    def create_data_access_policy(self, collection_name: str) -> bool:
        """Create data access policy for the collection"""
        policy_name = f"{collection_name}-data"

        policy = [
            {
                "Rules": [
                    {
                        "ResourceType": "collection",
                        "Resource": [f"collection/{collection_name}"],
                        "Permission": [
                            "aoss:CreateCollectionItems",
                            "aoss:UpdateCollectionItems",
                            "aoss:DescribeCollectionItems"
                        ]
                    },
                    {
                        "ResourceType": "index",
                        "Resource": [f"index/{collection_name}/*"],
                        "Permission": [
                            "aoss:CreateIndex",
                            "aoss:UpdateIndex",
                            "aoss:DescribeIndex",
                            "aoss:ReadDocument",
                            "aoss:WriteDocument"
                        ]
                    }
                ],
                "Principal": [self.principal_arn]
            }
        ]

        try:
            self.aoss_client.create_access_policy(
                name=policy_name,
                type='data',
                policy=json.dumps(policy)
            )
            print(f"✓ Created data access policy: {policy_name}")
            print(f"  Principal: {self.principal_arn}")
            return True
        except self.aoss_client.exceptions.ConflictException:
            print(f"  Data access policy already exists: {policy_name}")
            return True
        except Exception as e:
            print(f"✗ Error creating data access policy: {e}")
            return False

    def create_collection(self, collection_name: str) -> Optional[str]:
        """Create OpenSearch Serverless collection"""
        try:
            response = self.aoss_client.create_collection(
                name=collection_name,
                type='VECTORSEARCH',
                description='Vector search collection for movie recommendations'
            )

            collection_id = response['createCollectionDetail']['id']
            print(f"✓ Collection creation initiated: {collection_name}")
            print(f"  Collection ID: {collection_id}")

            # Wait for collection to become active
            print("  Waiting for collection to become active (this may take 2-3 minutes)...")

            max_attempts = 60
            for attempt in range(max_attempts):
                try:
                    status_response = self.aoss_client.batch_get_collection(
                        names=[collection_name]
                    )

                    if status_response['collectionDetails']:
                        collection = status_response['collectionDetails'][0]
                        status = collection['status']

                        if status == 'ACTIVE':
                            endpoint = collection['collectionEndpoint']
                            print(f"✓ Collection is active!")
                            print(f"  Endpoint: {endpoint}")
                            return endpoint
                        elif status == 'FAILED':
                            print(f"✗ Collection creation failed")
                            return None
                        else:
                            print(f"  Status: {status} (attempt {attempt + 1}/{max_attempts})")

                    time.sleep(5)

                except Exception as e:
                    print(f"  Checking status... (attempt {attempt + 1}/{max_attempts})")
                    time.sleep(5)

            print(f"✗ Timeout waiting for collection to become active")
            return None

        except self.aoss_client.exceptions.ConflictException:
            print(f"  Collection already exists: {collection_name}")
            # Get existing collection endpoint
            response = self.aoss_client.batch_get_collection(names=[collection_name])
            if response['collectionDetails']:
                endpoint = response['collectionDetails'][0]['collectionEndpoint']
                print(f"  Endpoint: {endpoint}")
                return endpoint
            return None

        except Exception as e:
            print(f"✗ Error creating collection: {e}")
            return None

    def create_index(self, endpoint: str, index_name: str) -> bool:
        """Create vector search index in the collection"""
        from opensearchpy import OpenSearch, RequestsHttpConnection
        from requests_aws4auth import AWS4Auth

        # AWS authentication
        credentials = boto3.Session().get_credentials()
        awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            self.region,
            'aoss',
            session_token=credentials.token
        )

        # Connect to OpenSearch
        host = endpoint.replace('https://', '')
        client = OpenSearch(
            hosts=[{'host': host, 'port': 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30
        )

        # Index mapping for vector search
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": Config.HNSW_EF_SEARCH
                }
            },
            "mappings": {
                "properties": {
                    "title": {"type": "text"},
                    "plot": {"type": "text"},
                    "genre": {"type": "keyword"},
                    "year": {"type": "integer"},
                    "rating": {"type": "float"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": Config.BEDROCK_EMBEDDING_DIMENSION,
                        "method": {
                            "name": "hnsw",
                            "space_type": "l2",
                            "engine": "nmslib",
                            "parameters": {
                                "m": Config.HNSW_M,
                                "ef_construction": Config.HNSW_EF_CONSTRUCTION
                            }
                        }
                    }
                }
            }
        }

        try:
            # Check if index exists
            if client.indices.exists(index=index_name):
                print(f"  Index already exists: {index_name}")
                return True

            # Create index
            client.indices.create(index=index_name, body=index_body)
            print(f"✓ Created index: {index_name}")
            print(f"  Vector dimensions: {Config.BEDROCK_EMBEDDING_DIMENSION}")
            print(f"  HNSW M: {Config.HNSW_M}")
            print(f"  HNSW ef_construction: {Config.HNSW_EF_CONSTRUCTION}")
            return True

        except Exception as e:
            print(f"✗ Error creating index: {e}")
            return False

    def setup(self, collection_name: str = 'movies-vector') -> Optional[str]:
        """Complete setup process"""
        print("=" * 70)
        print("OpenSearch Serverless Vector Search Setup")
        print("=" * 70)

        print(f"\nCollection name: {collection_name}")
        print(f"Region: {self.region}")
        print(f"Index: {Config.OPENSEARCH_INDEX}")

        print("\n" + "=" * 70)
        print("Step 1: Create Security Policies")
        print("=" * 70)

        if not self.create_encryption_policy(collection_name):
            return None

        if not self.create_network_policy(collection_name):
            return None

        if not self.create_data_access_policy(collection_name):
            return None

        print("\n" + "=" * 70)
        print("Step 2: Create Collection")
        print("=" * 70)

        endpoint = self.create_collection(collection_name)
        if not endpoint:
            return None

        print("\n" + "=" * 70)
        print("Step 3: Create Vector Index")
        print("=" * 70)

        if not self.create_index(endpoint, Config.OPENSEARCH_INDEX):
            return None

        print("\n" + "=" * 70)
        print("Setup Complete!")
        print("=" * 70)
        print(f"\n✓ Collection: {collection_name}")
        print(f"✓ Endpoint: {endpoint}")
        print(f"✓ Index: {Config.OPENSEARCH_INDEX}")
        print(f"\n📝 Add this to your .env file:")
        print(f"AOSS_VECTORSEARCH_ENDPOINT={endpoint}")

        return endpoint


def main():
    try:
        setup = OpenSearchSetup()
        endpoint = setup.setup()

        if endpoint:
            print("\n" + "=" * 70)
            print("Next Steps:")
            print("=" * 70)
            print("1. Add the endpoint to your .env file (see above)")
            print("2. Run: python ingest_data.py (to load movie data)")
            print("3. Run: streamlit run app.py (to start the search UI)")
        else:
            print("\n✗ Setup failed. Check the errors above.")
            return 1

        return 0

    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
