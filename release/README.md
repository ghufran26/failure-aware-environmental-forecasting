# Full output release

Upload the supplied **primna-tom-res.zip** as an asset on a GitHub Release. Do not put it in the ordinary Git history. Its size is 138,679,818 bytes. Keep its name and bytes unchanged so this checksum continues to identify the uploaded original.

```text
3969f726848a8241d7297e379838b94b604f653b8790d2f4aae07f72f6d0867e  primna-tom-res.zip
```

Verify the archive with:

```bash
python scripts/verify_output_archive.py /path/to/primna-tom-res.zip
```

During repository preparation, all 2,099 entries in `manifests/output_checksums.csv` matched their recorded lengths and SHA-256 values. macOS packaging metadata in the uploaded outer ZIP was excluded from the lightweight repository; the original ZIP is unchanged.

The archive includes detailed results, figures, tables, plot data, source, configuration, and manifests. It does not include the raw UCI datasets or resumable computation checkpoints. Official acquisition links are in `data/README.md`.

After publishing the release, add its actual URL to the main README. No repository URL, release URL, or DOI is fabricated in this package.
