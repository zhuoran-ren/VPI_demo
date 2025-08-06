/*
 *    This file is part of CasADi.
 *
 *    CasADi -- A symbolic framework for dynamic optimization.
 *    Copyright (C) 2010-2023 Joel Andersson, Joris Gillis, Moritz Diehl,
 *                            KU Leuven. All rights reserved.
 *    Copyright (C) 2011-2014 Greg Horn
 *
 *    CasADi is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    CasADi is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with CasADi; if not, write to the Free Software
 *    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

#ifndef CASADI_FMU2_HPP
#define CASADI_FMU2_HPP

#include "fmu_impl.hpp"

#include <fmi2Functions.h>

/// \cond INTERNAL

namespace casadi {

/** \brief Interface to a binary FMU, adhering to FMI version 2.0.

    \author Joel Andersson
    \date 2023

    \identifier{2at} */
class CASADI_EXPORT Fmu2 : public FmuInternal {
 public:
  // Constructor
  Fmu2(const std::string& name,
    const std::vector<std::string>& scheme_in, const std::vector<std::string>& scheme_out,
    const std::map<std::string, std::vector<size_t>>& scheme, const std::vector<std::string>& aux);

  /// Destructor
  ~Fmu2() override;

  /** \brief Get type name

      \identifier{2au} */
  std::string class_name() const override { return "Fmu2";}

  // Initialize
  void init(const DaeBuilderInternal* dae) override;

  /** \brief Initalize memory block

      \identifier{2do} */
  int init_mem(FmuMemory* m) const override;

  /** \brief Create memory block

      \identifier{2dp} */
  FmuMemory* alloc_mem(const FmuFunction& f) const override;

  /** \brief Free memory block

      \identifier{2du} */
  void free_mem(void *mem) const override;

  // Finalize
  void finalize() override;

  // Set C API functions
  void load_functions() override;

  // Variables used for initialization, by type
  std::vector<fmi2ValueReference> vr_real_, vr_integer_, vr_boolean_, vr_string_;
  std::vector<fmi2Real> init_real_;
  std::vector<fmi2Integer> init_integer_;
  std::vector<fmi2Boolean> init_boolean_;
  std::vector<std::string> init_string_;

  // Auxilliary variables, by type
  std::vector<std::string> vn_aux_real_, vn_aux_integer_, vn_aux_boolean_, vn_aux_string_;
  std::vector<fmi2ValueReference> vr_aux_real_, vr_aux_integer_, vr_aux_boolean_, vr_aux_string_;

  // Following members set in finalize

  // FMU C API function prototypes. Cf. FMI specification 2.0.2
  fmi2InstantiateTYPE* instantiate_;
  fmi2FreeInstanceTYPE* free_instance_;
  fmi2ResetTYPE* reset_;
  fmi2SetupExperimentTYPE* setup_experiment_;
  fmi2EnterInitializationModeTYPE* enter_initialization_mode_;
  fmi2ExitInitializationModeTYPE* exit_initialization_mode_;
  fmi2EnterContinuousTimeModeTYPE* enter_continuous_time_mode_;
  fmi2GetDerivativesTYPE* get_derivatives_;
  fmi2SetTimeTYPE* set_time_;
  fmi2GetRealTYPE* get_real_;
  fmi2SetRealTYPE* set_real_;
  fmi2GetBooleanTYPE* get_boolean_;
  fmi2SetBooleanTYPE* set_boolean_;
  fmi2GetIntegerTYPE* get_integer_;
  fmi2SetIntegerTYPE* set_integer_;
  fmi2GetStringTYPE* get_string_;
  fmi2SetStringTYPE* set_string_;
  fmi2GetDirectionalDerivativeTYPE* get_directional_derivative_;
  fmi2NewDiscreteStatesTYPE* new_discrete_states_;

  // Callback functions
  fmi2CallbackFunctions functions_;

  // Collection of variable values, all types
  struct Value {
    std::vector<fmi2Real> v_real;
    std::vector<fmi2Integer> v_integer;
    std::vector<fmi2Boolean> v_boolean;
    std::vector<std::string> v_string;
  };

  Value aux_value_;

  // Name of system, per the FMI specification
  std::string system_infix() const override;

  // New memory object
  void* instantiate() const override;

  // Free FMU instance
  void free_instance(void* instance) const override;

  // Reset solver
  int reset(void* instance);

  // Enter initialization mode
  int enter_initialization_mode(void* instance) const override;

  // Exit initialization mode
  int exit_initialization_mode(void* instance) const override;

  // Enter continuous-time mode
  int enter_continuous_time_mode(void* instance) const override;

  // Update discrete states
  int update_discrete_states(void* instance, EventMemory* eventmem) const override;

  int get_derivatives(void* instance, double* derivatives, size_t nx) const override;

  // Set real values
  int set_real(void* instance, const unsigned int* vr, size_t n_vr,
    const double* values, size_t n_values) const override;

  // Get/evaluate real values
  int get_real(void* instance, const unsigned int* vr, size_t n_vr,
    double* values, size_t n_values) const override;

  // Forward mode AD
  int get_directional_derivative(void* instance, const unsigned int* vr_out, size_t n_out,
    const unsigned int* vr_in, size_t n_in, const double* seed, size_t n_seed,
    double* sensitivity, size_t n_sensitivity) const override;

  // Copy values set in DaeBuilder to FMU
  int set_values(void* instance) const override;

  // Retrieve auxilliary variables from FMU
  int get_aux(void* instance) override;

  // Retrieve auxilliary variables from FMU, implementation
  int get_aux_impl(void* instance, Value& aux_value) const;

  /** \brief Get stats

      \identifier{2av} */
  void get_stats(FmuMemory* m, Dict* stats,
    const std::vector<std::string>& name_in, const InputStruct* in) const override;

  // Process message
  static void logger(fmi2ComponentEnvironment componentEnvironment,
    fmi2String instanceName,
    fmi2Status status,
    fmi2String category,
    fmi2String message, ...);

  void serialize_body(SerializingStream& s) const override;

  static Fmu2* deserialize(DeserializingStream& s);

  protected:
    explicit Fmu2(DeserializingStream& s);
};

} // namespace casadi

/// \endcond

#endif // CASADI_FMU2_HPP
