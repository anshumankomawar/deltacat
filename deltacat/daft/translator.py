from daft import Expression
from daft.daft import Pushdowns
import pyarrow as pa
from typing import Any, Callable, Dict, List
from daft.expressions import Expression as DaftExpression
from daft.io.pushdowns import (
    Literal as DaftLiteral,
    Reference as DaftReference,
    TermVisitor,
)
from daft.expressions.visitor import PredicateVisitor 
from daft.logical.schema import DataType

from deltacat.storage.model.expression import (
    Expression,
    Reference,
    Literal,
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
    GreaterThanEqual,
    LessThanEqual,
    And,
    Or,
    Not,
    IsNull,
)
from deltacat.storage.model.scan.push_down import PartitionFilter, Pushdown, RowFilter, ColumnFilter


def translate_pushdown(pushdown: Pushdowns) -> Pushdown:
    """
    Helper method to translate a Daft ScanPushdowns object into a Deltacat Pushdown.

    Args:
        pushdown: Daft ScanPushdowns object

    Returns:
        Pushdown: Deltacat Pushdown object with translated filters
    """
    translator = DaftToDeltacatVisitor()

    partition_filters = None
    if pushdown.partition_filters is not None:
        daft_expr = DaftExpression._from_pyexpr(pushdown.partition_filters)
        partition_filters = PartitionFilter.of(translator.visit(daft_expr))

    filters = None
    if pushdown.filters is not None:
        daft_expr = DaftExpression._from_pyexpr(pushdown.filters)
        # TODO: support deltacat row filters
        # filters = RowFilter.of(translator.visit(daft_expr))

    columns = None
    limit = None
    
    return Pushdown.of(
        partition_filter=partition_filters,
        column_filter=columns,
        row_filter=filters,
        limit=limit,
    )


class DaftToDeltacatVisitor(PredicateVisitor[Expression]):
    """PredicateVisitor implementation to translate Daft Expressions into Deltacat Expressions"""

    def visit_col(self, name: str) -> Expression:
        return Reference.of(name)

    def visit_lit(self, value: Any) -> Expression:
        return Literal.of(value)

    def visit_cast(self, expr: DaftExpression, dtype: DataType) -> Expression:
        # deltacat expressions do not support explicit casting
        # pyarrow should handle any type casting 
        return self.visit(expr)

    def visit_alias(self, expr: DaftExpression, alias: str) -> Expression:
        return self.visit(expr)

    def visit_function(self, name: str, args: List[DaftExpression]) -> Expression:
        # TODO: Add Deltacat expression function support
        raise ValueError("Function not supported")

    def visit_and(self, left: DaftExpression, right: DaftExpression) -> Expression:
        """Visit an 'and' expression."""
        return And.of(self.visit(left), self.visit(right))

    def visit_or(self, left: DaftExpression, right: DaftExpression) -> Expression:
        """Visit an 'or' expression."""
        return Or.of(self.visit(left), self.visit(right))

    def visit_not(self, expr: DaftExpression) -> Expression:
        """Visit a 'not' expression."""
        return Not.of(self.visit(expr))

    def visit_equal(self, left: DaftExpression, right: DaftExpression) -> Expression:
        """Visit an 'equals' comparison predicate."""
        return Equal.of(self.visit(left), self.visit(right))

    def visit_not_equal(self, left: DaftExpression, right: DaftExpression) -> Expression:
        """Visit a 'not equals' comparison predicate."""
        return NotEqual.of(self.visit(left), self.visit(right))

    def visit_less_than(self, left: DaftExpression, right: DaftExpression) -> Expression:
        """Visit a 'less than' comparison predicate."""
        return LessThan.of(self.visit(left), self.visit(right))

    def visit_less_than_or_equal(self, left: DaftExpression, right: DaftExpression) -> Expression:
        """Visit a 'less than or equal' comparison predicate."""
        return LessThanEqual.of(self.visit(left), self.visit(right))

    def visit_greater_than(self, left: DaftExpression, right: DaftExpression) -> Expression:
        """Visit a 'greater than' comparison predicate."""
        return GreaterThan.of(self.visit(left), self.visit(right))

    def visit_greater_than_or_equal(self, left: DaftExpression, right: DaftExpression) -> Expression:
        """Visit a 'greater than or equal' comparison predicate."""
        return GreaterThanEqual.of(self.visit(left), self.visit(right))

    def visit_between(self, expr: DaftExpression, lower: DaftExpression, upper: DaftExpression) -> Expression:
        """Visit a 'between' predicate."""
        # Implement BETWEEN as lower <= expr <= upper
        lower_bound = LessThanEqual.of(self.visit(lower), self.visit(expr))
        upper_bound = LessThanEqual.of(self.visit(expr), self.visit(upper))
        return And.of(lower_bound, upper_bound)

    def visit_is_in(self, expr: DaftExpression, items: list[DaftExpression]) -> Expression:
        """Visit an 'is_in' predicate."""
        # For empty list, return false literal
        if not items:
            return Literal(pa.scalar(False))
        
        # Implement IN as a series of equality checks combined with OR
        visited_expr = self.visit(expr)
        equals_exprs = [Equal.of(visited_expr, self.visit(item)) for item in items]
        
        # Combine with OR
        result = equals_exprs[0]
        for eq_expr in equals_exprs[1:]:
            result = Or.of(result, eq_expr)
        
        return result

    def visit_is_null(self, expr: DaftExpression) -> Expression:
        """Visit an 'is_null' predicate."""
        return IsNull.of(self.visit(expr))

    def visit_not_null(self, expr: DaftExpression) -> Expression:
        """Visit an 'not_null' predicate."""
        # NOT NULL is implemented as NOT(IS NULL)
        return Not.of(IsNull.of(self.visit(expr)))
