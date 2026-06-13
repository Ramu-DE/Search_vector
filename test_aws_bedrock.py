#!/usr/bin/env python3
"""
Test AWS Bedrock Access and Embedding Generation
Verifies your AWS credentials and Bedrock permissions
"""

import boto3
import json
from typing import List, Dict, Any


def test_aws_credentials():
    """Test basic AWS credentials"""
    print("=" * 60)
    print("Testing AWS Credentials")
    print("=" * 60)

    try:
        sts = boto3.client('sts', region_name='us-west-2')
        identity = sts.get_caller_identity()

        print(f"✓ AWS credentials are valid")
        print(f"  Account: {identity['Account']}")
        print(f"  User ARN: {identity['Arn']}")
        print(f"  User ID: {identity['UserId']}")
        return True

    except Exception as e:
        print(f"✗ AWS credential error: {e}")
        return False


def test_bedrock_access():
    """Test Bedrock service access"""
    print("\n" + "=" * 60)
    print("Testing Bedrock Access")
    print("=" * 60)

    try:
        bedrock = boto3.client('bedrock', region_name='us-west-2')

        # List available models
        response = bedrock.list_foundation_models()

        # Filter for embedding models
        embedding_models = [
            model for model in response['modelSummaries']
            if 'embed' in model['modelId'].lower()
        ]

        print(f"✓ Bedrock access verified")
        print(f"  Available embedding models: {len(embedding_models)}")

        print("\n  Titan Embedding Models:")
        for model in embedding_models:
            if 'titan' in model['modelId'].lower():
                print(f"    • {model['modelId']}")
                print(f"      Name: {model['modelName']}")

        return True, embedding_models

    except Exception as e:
        print(f"✗ Bedrock access error: {e}")
        return False, []


def test_embedding_generation(model_id: str = "amazon.titan-embed-text-v2:0"):
    """Test embedding generation with Bedrock"""
    print("\n" + "=" * 60)
    print(f"Testing Embedding Generation: {model_id}")
    print("=" * 60)

    try:
        bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-west-2')

        # Test text
        test_text = "A thrilling space adventure movie"

        # Create request body
        body = json.dumps({
            "inputText": test_text
        })

        # Invoke model
        print(f"  Generating embedding for: '{test_text}'")
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=body,
            contentType='application/json',
            accept='application/json'
        )

        # Parse response
        response_body = json.loads(response['body'].read())
        embedding = response_body.get('embedding', [])

        print(f"✓ Embedding generated successfully")
        print(f"  Dimensions: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")
        print(f"  Model ID used: {model_id}")

        return True, len(embedding)

    except Exception as e:
        print(f"✗ Embedding generation error: {e}")
        return False, 0


def test_config_compatibility():
    """Test if config.py settings match available models"""
    print("\n" + "=" * 60)
    print("Testing Config Compatibility")
    print("=" * 60)

    try:
        from config import Config

        print(f"  Configured model: {Config.BEDROCK_MODEL_ID}")
        print(f"  Expected dimensions: {Config.BEDROCK_EMBEDDING_DIMENSION}")
        print(f"  Region: {Config.AWS_REGION}")

        # Test with the exact model ID from config
        # Try both with and without version suffix
        model_variants = [
            Config.BEDROCK_MODEL_ID,
            f"{Config.BEDROCK_MODEL_ID}:0",
            "amazon.titan-embed-text-v2:0"
        ]

        success = False
        actual_dimensions = 0

        for model_id in model_variants:
            print(f"\n  Trying: {model_id}")
            result, dims = test_embedding_generation(model_id)
            if result:
                success = True
                actual_dimensions = dims

                # Check dimension match
                if dims == Config.BEDROCK_EMBEDDING_DIMENSION:
                    print(f"  ✓ Dimensions match config ({dims})")
                else:
                    print(f"  ⚠️  Dimension mismatch!")
                    print(f"     Config expects: {Config.BEDROCK_EMBEDDING_DIMENSION}")
                    print(f"     Model returns: {dims}")
                break

        if not success:
            print(f"  ✗ Could not generate embeddings with configured model")
            return False

        return True

    except ImportError as e:
        print(f"  ✗ Could not import config: {e}")
        return False


def check_opensearch_endpoint():
    """Check if OpenSearch endpoint is configured"""
    print("\n" + "=" * 60)
    print("Checking OpenSearch Configuration")
    print("=" * 60)

    try:
        from config import Config

        if Config.OPENSEARCH_ENDPOINT:
            print(f"✓ OpenSearch endpoint configured")
            print(f"  Endpoint: {Config.OPENSEARCH_ENDPOINT}")
            print(f"  Index: {Config.OPENSEARCH_INDEX}")
            return True
        else:
            print(f"⚠️  OpenSearch endpoint not configured")
            print(f"  Set AOSS_VECTORSEARCH_ENDPOINT environment variable")
            return False

    except Exception as e:
        print(f"✗ Error checking config: {e}")
        return False


def main():
    print("\n" + "╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "AWS BEDROCK TEST SUITE" + " " * 21 + "║")
    print("╚" + "=" * 58 + "╝\n")

    results = {}

    # Test 1: AWS Credentials
    results['credentials'] = test_aws_credentials()

    # Test 2: Bedrock Access
    if results['credentials']:
        results['bedrock'], models = test_bedrock_access()
    else:
        print("\n⚠️  Skipping Bedrock tests (credentials failed)")
        results['bedrock'] = False

    # Test 3: Config Compatibility
    if results.get('bedrock', False):
        results['config'] = test_config_compatibility()

    # Test 4: OpenSearch Configuration
    results['opensearch'] = check_opensearch_endpoint()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"✓ AWS Credentials: {'PASS' if results.get('credentials') else 'FAIL'}")
    print(f"✓ Bedrock Access: {'PASS' if results.get('bedrock') else 'FAIL'}")
    print(f"✓ Config Compatibility: {'PASS' if results.get('config') else 'FAIL'}")
    print(f"{'✓' if results.get('opensearch') else '⚠️ '} OpenSearch Config: {'CONFIGURED' if results.get('opensearch') else 'NOT SET'}")

    if all([results.get('credentials'), results.get('bedrock'), results.get('config')]):
        print("\n🎉 All critical tests passed! Your AWS Bedrock setup is working.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

    print("=" * 60)


if __name__ == "__main__":
    main()
