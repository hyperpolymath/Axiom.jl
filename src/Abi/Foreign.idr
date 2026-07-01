-- SPDX-License-Identifier: MPL-2.0
||| Axiom.jl Foreign Function Interface Declarations
|||
||| ILLUSTRATIVE SCAFFOLD -- NOT A PROOF OF AXIOM'S PRODUCTION FFI BOUNDARY.
||| The symbols declared here (axiom_init, axiom_process, axiom_free,
||| axiom_register_callback, axiom_invoke_callback, etc.) are a generic
||| lifecycle/callback surface for the hyperpolymath ABI-FFI pattern. They
||| use concrete Axiom naming but do NOT correspond to Axiom's real Zig
||| kernel exports (axiom_matmul, axiom_relu, axiom_conv2d, and the rest of
||| the ~36 real exports in `ffi/zig/src/main.zig` / `zig/src/axiom.zig`
||| consumed by `src/backends/zig_ffi.jl`). No `libaxiom` implementing
||| exactly this surface is built by this repository.
|||
||| The production FFI boundary is Julia <-> Zig via `src/backends/zig_ffi.jl`
||| and `ffi/zig/`; see `ABI-FFI-README.md` for the authoritative split.
||| Connecting this scaffold's declarations to the real Zig exports is
||| tracked future work (see ROADMAP note in `src/Abi/Types.idr`).
|||
||| This module declares C ABI symbols used by the Idris2 ABI scaffold.
||| Symbol names are concrete for Axiom (`axiom_*`, `libaxiom`).

module Abi.Foreign

import Abi.Types
import Abi.Layout
import Data.Maybe

%default total

--------------------------------------------------------------------------------
-- Library Lifecycle
--------------------------------------------------------------------------------

||| Initialize Axiom runtime state.
export
%foreign "C:axiom_init, libaxiom"
prim__init : PrimIO Bits64

||| Safe initialization wrapper.
export
init : IO (Maybe Handle)
init = do
  ptr <- primIO prim__init
  pure (createHandle ptr)

||| Dispose runtime state.
export
%foreign "C:axiom_free, libaxiom"
prim__free : Bits64 -> PrimIO ()

||| Safe cleanup wrapper.
export
free : Handle -> IO ()
free h = primIO (prim__free (handlePtr h))

--------------------------------------------------------------------------------
-- Core Operations
--------------------------------------------------------------------------------

||| Run a core operation; returns `Result` code.
export
%foreign "C:axiom_process, libaxiom"
prim__process : Bits64 -> Bits32 -> PrimIO Bits32

||| Safe process wrapper.
export
process : Handle -> Bits32 -> IO (Either Result Bits32)
process h input = do
  code <- primIO (prim__process (handlePtr h) input)
  pure $ case intToResult code of
    Just Ok => Right input
    Just err => Left err
    Nothing => Left Error

--------------------------------------------------------------------------------
-- String Operations
--------------------------------------------------------------------------------

||| Convert C string pointer to Idris string.
export
%foreign "support:idris2_getString, libidris2_support"
prim__getString : Bits64 -> String

||| Free heap string returned by runtime.
export
%foreign "C:axiom_free_string, libaxiom"
prim__freeString : Bits64 -> PrimIO ()

||| Request a string result.
export
%foreign "C:axiom_get_string, libaxiom"
prim__getResult : Bits64 -> PrimIO Bits64

||| Safe string fetch wrapper.
export
getString : Handle -> IO (Maybe String)
getString h = do
  ptr <- primIO (prim__getResult (handlePtr h))
  if ptr == 0
    then pure Nothing
    else do
      let s = prim__getString ptr
      primIO (prim__freeString ptr)
      pure (Just s)

--------------------------------------------------------------------------------
-- Array/Buffer Operations
--------------------------------------------------------------------------------

||| Process buffer payload.
export
%foreign "C:axiom_process_array, libaxiom"
prim__processArray : Bits64 -> Bits64 -> Bits32 -> PrimIO Bits32

||| Safe array processor.
export
processArray : Handle -> Bits64 -> Bits32 -> IO (Either Result ())
processArray h buf len = do
  code <- primIO (prim__processArray (handlePtr h) buf len)
  pure $ case intToResult code of
    Just Ok => Right ()
    Just err => Left err
    Nothing => Left Error

--------------------------------------------------------------------------------
-- Error Handling
--------------------------------------------------------------------------------

||| Last error pointer.
export
%foreign "C:axiom_last_error, libaxiom"
prim__lastError : PrimIO Bits64

||| Safe last-error wrapper.
export
lastError : IO (Maybe String)
lastError = do
  ptr <- primIO prim__lastError
  if ptr == 0 then pure Nothing else pure (Just (prim__getString ptr))

||| Human-readable description for typed result.
export
errorDescription : Result -> String
errorDescription Ok = "Success"
errorDescription Error = "Generic error"
errorDescription InvalidParam = "Invalid parameter"
errorDescription OutOfMemory = "Out of memory"
errorDescription NullPointer = "Null pointer"

--------------------------------------------------------------------------------
-- Version Information
--------------------------------------------------------------------------------

||| Runtime version string pointer.
export
%foreign "C:axiom_version, libaxiom"
prim__version : PrimIO Bits64

||| Version string.
export
version : IO String
version = do
  ptr <- primIO prim__version
  pure (prim__getString ptr)

||| Build info pointer.
export
%foreign "C:axiom_build_info, libaxiom"
prim__buildInfo : PrimIO Bits64

||| Build info string.
export
buildInfo : IO String
buildInfo = do
  ptr <- primIO prim__buildInfo
  pure (prim__getString ptr)

--------------------------------------------------------------------------------
-- Callback Registration (C -> Idris bridge point)
--------------------------------------------------------------------------------

||| Callback type used at the Idris layer.
public export
Callback : Type
Callback = Bits64 -> Bits32 -> Bits32

||| Register a callback function pointer (raw C pointer).
export
%foreign "C:axiom_register_callback, libaxiom"
prim__registerCallback : Bits64 -> Bits64 -> PrimIO Bits32

||| Safe callback-pointer registration wrapper.
export
registerCallbackPtr : Handle -> Bits64 -> IO (Either Result ())
registerCallbackPtr h cbPtr = do
  code <- primIO (prim__registerCallback (handlePtr h) cbPtr)
  pure $ case intToResult code of
    Just Ok => Right ()
    Just err => Left err
    Nothing => Left Error

||| Invoke a previously registered callback via runtime (bidirectional bridge).
||| `ctx` is user-defined callback context.
||| `outPtr` points to writable memory for the callback result (`Bits32`).
export
%foreign "C:axiom_invoke_callback, libaxiom"
prim__invokeCallback : Bits64 -> Bits64 -> Bits32 -> Bits64 -> PrimIO Bits32

||| Safe callback invocation wrapper using a raw output pointer.
export
invokeCallbackPtr : Handle -> Bits64 -> Bits32 -> Bits64 -> IO (Either Result ())
invokeCallbackPtr h ctx input outPtr = do
  code <- primIO (prim__invokeCallback (handlePtr h) ctx input outPtr)
  pure $ case intToResult code of
    Just Ok => Right ()
    Just err => Left err
    Nothing => Left Error

--------------------------------------------------------------------------------
-- Utility
--------------------------------------------------------------------------------

||| Query initialization state.
export
%foreign "C:axiom_is_initialized, libaxiom"
prim__isInitialized : Bits64 -> PrimIO Bits32

||| Safe initialization-state wrapper.
export
isInitialized : Handle -> IO Bool
isInitialized h = do
  code <- primIO (prim__isInitialized (handlePtr h))
  pure (code /= 0)
