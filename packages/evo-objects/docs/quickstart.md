# Getting started

## Basic usage

Just want to get going? See [the evo-sdk-common documentation](https://pypi.org/project/evo-sdk-common/) for information on how to authenticate, then select the organisation, hub and workspace that you would like to use.

### Interacting with the Geoscience Object API

`evo.objects.ObjectAPIClient` requires an `evo.common.Environment` and an
`evo.common.APIConnector`. Workspace objects have a `get_environment()` method that returns an
environment object, which can be used by one or more service clients to interact with different services using the same
organization and workspace. The hub connector that was used for workspace discovery can be reused for interacting
with services.

``` python
service_client = ObjectAPIClient(workspace_env, hub_connector)
service_health = await service_client.get_service_health()
service_health.raise_for_status()

# The data client is an optional utility that provides helpers for uploading and downloading
# parquet data via pyarrow.Table objects.
data_client = service_client.get_data_client(manager.cache)
...
```

Listing objects is simple, just call the `ObjectAPIClient.list_objects()` method.

``` python
offset = 0
while True:
    page = await service_client.list_objects(offset=offset, limit=10)
    if offset == 0:
        print(f"Found {page.total} object{'' if page.total == 1 else 's'}")
    for object in page:
        print(f"{object.path}: <{object.schema_id}> ({object.id})")

    if page.is_last:
        break
    else:
        offset = page.next_offset
```

You can also get a list of all objects. Internally, this recursively calls the `list_objects()` method until all objects are fetched.

Check out the other methods  on the `ObjectAPIClient` for more details on how to upload and download objects, or get object versions.
