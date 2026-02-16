||| Axiom.jl ABI Type Definitions
|||
||| This module defines the core ABI-facing types used by the Idris2
||| specification layer for Axiom.jl.

module Abi.Types

import Data.Bits
import Data.Maybe

%default total

--------------------------------------------------------------------------------
-- Platform
--------------------------------------------------------------------------------

||| Supported target platforms.
public export
data Platform = Linux | Windows | MacOS | BSD | WASM

||| Default platform used by the lightweight ABI scaffold.
public export
thisPlatform : Platform
thisPlatform = Linux

--------------------------------------------------------------------------------
-- Result Codes
--------------------------------------------------------------------------------

||| C-compatible result codes.
public export
data Result = Ok | Error | InvalidParam | OutOfMemory | NullPointer

||| Convert a result into its ABI integer representation.
public export
resultToInt : Result -> Bits32
resultToInt Ok = 0
resultToInt Error = 1
resultToInt InvalidParam = 2
resultToInt OutOfMemory = 3
resultToInt NullPointer = 4

||| Convert a C integer code back into a typed result.
public export
intToResult : Bits32 -> Maybe Result
intToResult 0 = Just Ok
intToResult 1 = Just Error
intToResult 2 = Just InvalidParam
intToResult 3 = Just OutOfMemory
intToResult 4 = Just NullPointer
intToResult _ = Nothing

--------------------------------------------------------------------------------
-- Opaque Handle
--------------------------------------------------------------------------------

||| Opaque handle wrapper used across FFI boundaries.
public export
record Handle where
  constructor MkHandle
  ptr : Bits64

||| Construct a handle only when pointer is non-null.
public export
createHandle : Bits64 -> Maybe Handle
createHandle 0 = Nothing
createHandle ptr = Just (MkHandle ptr)

||| Extract raw pointer value.
public export
handlePtr : Handle -> Bits64
handlePtr h = h.ptr

--------------------------------------------------------------------------------
-- Platform-Dependent Aliases
--------------------------------------------------------------------------------

||| C int mapping.
public export
CInt : Platform -> Type
CInt Linux = Bits32
CInt Windows = Bits32
CInt MacOS = Bits32
CInt BSD = Bits32
CInt WASM = Bits32

||| C size_t mapping.
public export
CSize : Platform -> Type
CSize Linux = Bits64
CSize Windows = Bits64
CSize MacOS = Bits64
CSize BSD = Bits64
CSize WASM = Bits32

||| Pointer width (in bits).
public export
ptrSize : Platform -> Nat
ptrSize Linux = 64
ptrSize Windows = 64
ptrSize MacOS = 64
ptrSize BSD = 64
ptrSize WASM = 32

--------------------------------------------------------------------------------
-- Lightweight Size/Alignment Witnesses
--------------------------------------------------------------------------------

||| Witness that a type has a given size.
public export
data HasSize : Type -> Nat -> Type where
  SizeProof : HasSize t n

||| Witness that a type has a given alignment.
public export
data HasAlignment : Type -> Nat -> Type where
  AlignProof : HasAlignment t n

--------------------------------------------------------------------------------
-- Example ABI Struct
--------------------------------------------------------------------------------

||| Example C-compatible struct used by the ABI scaffold.
public export
record ExampleStruct where
  constructor MkExampleStruct
  field1 : Bits32
  field2 : Bits64
  field3 : Double

||| Example size witness (layout-dependent, fixed for scaffold docs).
public export
exampleStructSize : HasSize ExampleStruct 24
exampleStructSize = SizeProof

||| Example alignment witness.
public export
exampleStructAlign : HasAlignment ExampleStruct 8
exampleStructAlign = AlignProof

--------------------------------------------------------------------------------
-- Verification Hooks
--------------------------------------------------------------------------------

namespace Verify

  ||| Placeholder verification hook for ABI size checks.
  export
  verifySizes : IO ()
  verifySizes = putStrLn "Axiom ABI size witnesses loaded"

  ||| Placeholder verification hook for ABI alignment checks.
  export
  verifyAlignments : IO ()
  verifyAlignments = putStrLn "Axiom ABI alignment witnesses loaded"
