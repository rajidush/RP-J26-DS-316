"""
Component 2 (Hate-Speech Detection) storage -- PLACEHOLDER.

Not implemented until Component 2's output schema is final
(docs/interface-contracts/comp2_to_comp3.schema.json). Follow db/comp3.py and
the checklist in db/__init__.py:

    class Comp2Output(BaseModel): ...            # Pydantic mirror of the contract

    class Comp2Detection(Base):
        __tablename__ = "comp2_detections"
        id = mapped_column(Integer, primary_key=True, autoincrement=True)
        session_id = mapped_column(String(128), index=True)   # shared key, not unique
        ...

    def save_comp2_output(data: dict) -> int: ...

Then uncomment `from db import comp2` in db.base.init_db().
"""
