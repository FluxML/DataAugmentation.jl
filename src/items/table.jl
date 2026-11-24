"""
    TabularItem(data, columns)

Item that represents one row of a table. `data` holds the row's values (for
example a `NamedTuple`) and `columns` the corresponding column names.

Used by the row transformations [`NormalizeRow`](@ref), [`FillMissing`](@ref) and
[`Categorify`](@ref).
"""
struct TabularItem{T} <: Item
    data::T
    columns
end
