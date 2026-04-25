from .base_class import BaseClass  # noqa: TID252  # intentional relative import for regression test


class SubClass(BaseClass):
    def value(self):
        return self.base_value() + "+sub"
