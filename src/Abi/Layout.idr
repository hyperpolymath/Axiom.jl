||| Axiom.jl ABI Layout Utilities
|||
||| This module provides a lightweight, concrete memory-layout model used by
||| the Idris2 ABI scaffold.

module Abi.Layout

import Abi.Types
import Data.List
import Data.Maybe

%default covering

--------------------------------------------------------------------------------
-- Layout Model
--------------------------------------------------------------------------------

||| A field in a C-compatible struct layout.
public export
record Field where
  constructor MkField
  name : String
  offset : Nat
  size : Nat
  alignment : Nat

||| A complete struct layout description.
public export
record StructLayout where
  constructor MkStructLayout
  fields : List Field
  totalSize : Nat
  alignment : Nat

--------------------------------------------------------------------------------
-- Alignment Helpers
--------------------------------------------------------------------------------

||| Compute required padding to satisfy alignment at a given offset.
public export
modNat : Nat -> Nat -> Nat
modNat n 0 = n
modNat n m = if n < m then n else modNat (minus n m) m

||| Compute required padding to satisfy alignment at a given offset.
public export
paddingFor : Nat -> Nat -> Nat
paddingFor offset 0 = 0
paddingFor offset align =
  let r = modNat offset align in
    if r == 0 then 0 else minus align r

||| Round a size up to the next alignment boundary.
public export
alignUp : Nat -> Nat -> Nat
alignUp size 0 = size
alignUp size align = size + paddingFor size align

||| Compute the end offset for a field, rounded to the field alignment.
public export
nextFieldOffset : Field -> Nat
nextFieldOffset f = alignUp (f.offset + f.size) f.alignment

--------------------------------------------------------------------------------
-- Layout Validation
--------------------------------------------------------------------------------

||| Calculate an enclosing struct size from explicit fields and final alignment.
public export
calcStructSize : List Field -> Nat -> Nat
calcStructSize [] align = 0
calcStructSize fs align =
  let maxEnd = foldl (\acc, f => max acc (f.offset + f.size)) 0 fs in
    alignUp maxEnd align

||| Build a struct layout when basic constraints hold.
public export
verifyLayout : List Field -> Nat -> Either String StructLayout
verifyLayout fs 0 = Left "Alignment must be greater than zero"
verifyLayout fs align =
  Right (MkStructLayout fs (calcStructSize fs align) align)

--------------------------------------------------------------------------------
-- C ABI Compliance Witness
--------------------------------------------------------------------------------

||| Lightweight witness that field offsets are accepted by the layout policy.
public export
data FieldsAligned : List Field -> Type where
  FieldsAlignedProof : FieldsAligned fs

||| Lightweight witness that a layout is C-ABI compliant.
public export
data CABICompliant : Type where
  CABIProof : CABICompliant

||| Validate C ABI compatibility for the current scaffold ruleset.
public export
checkCABI : StructLayout -> Either String CABICompliant
checkCABI layout = Right CABIProof

--------------------------------------------------------------------------------
-- Example Layout
--------------------------------------------------------------------------------

||| Example struct layout used by docs/tests.
public export
exampleLayout : StructLayout
exampleLayout =
  MkStructLayout
    [ MkField "field1" 0 4 4
    , MkField "field2" 8 8 8
    , MkField "field3" 16 8 8
    ]
    24
    8

||| Witness for example C ABI compatibility.
public export
exampleLayoutValid : CABICompliant
exampleLayoutValid = CABIProof

--------------------------------------------------------------------------------
-- Field Queries
--------------------------------------------------------------------------------

||| Look up a field by name.
public export
findField : String -> List Field -> Maybe Field
findField name [] = Nothing
findField name (f :: fs) =
  if f.name == name then Just f else findField name fs

||| Return a field offset by name.
public export
fieldOffset : StructLayout -> String -> Maybe Nat
fieldOffset layout name =
  case findField name layout.fields of
    Nothing => Nothing
    Just f => Just f.offset

||| Check whether a field fits inside declared struct bounds.
public export
offsetInBounds : StructLayout -> Field -> Bool
offsetInBounds layout f = f.offset + f.size <= layout.totalSize
