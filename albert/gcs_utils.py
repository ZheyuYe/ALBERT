def explicit(gcs_json_file):
    from google.cloud import storage
    from google.colab import auth
    auth.authenticate_user()
    # Explicitly use service account credentials by specifying the private key
    # file.
    storage_client = storage.Client.from_service_account_json(gcs_json_file)

    # Make an authenticated API request
    buckets = list(storage_client.list_buckets())
    print(buckets)
