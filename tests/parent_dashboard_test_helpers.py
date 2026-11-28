def follow_page_link(at, area, title):
    link = next(link.proto for link in area.get("page_link") if link.proto.label == title)
    # AppTest.switch_page hashes the filename, but this app sets explicit URL paths.
    at._page_hash = link.page_script_hash
    return at.run()
