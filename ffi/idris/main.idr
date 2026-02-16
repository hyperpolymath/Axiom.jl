module Main

import System.FFI

%foreign "C:axiom_init"
export
axiom_init : IO (Ptr ())
axiom_init = do
    handle_ptr <- calloc 1 (sizeof Ptr)
    if handle_ptr == null then
        pure null
    else
        pure handle_ptr

%foreign "C:axiom_free"
export
axiom_free : Ptr () -> IO ()
axiom_free handle = free handle

%foreign "C:axiom_process"
export
axiom_process : Ptr () -> Bits32 -> IO Int
axiom_process handle input = pure 0

%foreign "C:axiom_get_string"
export
axiom_get_string : Ptr () -> IO (Ptr CChar)
axiom_get_string handle = newCString "Example String"

%foreign "C:axiom_free_string"
export
axiom_free_string : Ptr CChar -> IO ()
axiom_free_string str = free str

%foreign "C:axiom_process_array"
export
axiom_process_array : Ptr () -> Ptr Bits8 -> Bits32 -> IO Int
axiom_process_array handle buffer len = pure 0

%foreign "C:axiom_last_error"
export
axiom_last_error : IO (Ptr CChar)
axiom_last_error = newCString "No error"

%foreign "C:axiom_version"
export
axiom_version : IO (Ptr CChar)
axiom_version = newCString "0.1.0"

%foreign "C:axiom_build_info"
export
axiom_build_info : IO (Ptr CChar)
axiom_build_info = newCString "Axiom built with Idris2"

%foreign "C:axiom_register_callback"
export
axiom_register_callback : Ptr () -> FunPtr (Bits64 -> Bits32 -> IO Bits32) -> IO Int
axiom_register_callback handle callback = pure 0

%foreign "C:axiom_is_initialized"
export
axiom_is_initialized : Ptr () -> IO Bits32
axiom_is_initialized handle = pure 1

main : IO ()
main = putStrLn "This is the FFI module, not an executable."
