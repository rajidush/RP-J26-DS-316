"""
Component 1 (Screen Monitoring) storage -- PLACEHOLDER.

Not implemented until Component 1's output schema is final
(docs/interface-contracts/comp1_to_comp3.schema.json). Follow db/comp3.py and
the checklist in db/__init__.py:

    class Comp1Output(BaseModel): ...            # Pydantic mirror of the contract

    class Comp1Event(Base):
        __tablename__ = "comp1_events"
        id = mapped_column(Integer, primary_key=True, autoincrement=True)
        session_id = mapped_column(String(128), index=True)   # shared key, not unique
        ...

    def save_comp1_output(data: dict) -> int: ...

Then uncomment `from db import comp1` in db.base.init_db().
"""
