function select_time_varying_function(inp::String, default_const=1.0)
   functions = Dict(
       "sin" => sin,
       "cos" => cos,
       "exp" => exp,
       "log" => log,
       "constant" => t -> default_const
   )
   return functions[inp]
end