/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

 /*!
 将高层的 Relay 中间表示（IRModule）转换为目标设备可执行的代码 / 模块
  * \file relay/backend/build_module.cc
  * \brief Code generation for TVM's graph executor.
  */
#include <tvm/driver/driver_api.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/memory_pools.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/executor.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/qnn/transform.h>
#include <tvm/relay/runtime.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/device_api.h>
#include <tvm/target/compilation_config.h>

#include <memory>

#include "../../driver/internal_driver_api.h"
#include "../../target/func_registry_generator.h"
#include "../../target/metadata_module.h"
#include "../../target/source/codegen_source_base.h"
#include "te_compiler.h"
#include "utils.h"

namespace tvm {
  namespace relay {
    namespace transform {
      Pass LabelOps();
      }
    namespace backend {

      using namespace tvm::relay::transform;

      /*!
       * \brief Output of building module
       */
      struct BuildOutput {
        std::string graph_json;
        runtime::Module mod;          // 编译后的TVM运行时模块（含机器码/算子实现）
        std::unordered_map<std::string, tvm::runtime::NDArray> params;
        };

      /*!
       * \brief Executor代码生成器的抽象基类（接口）
       * 它封装了不同执行器（如Graph, AOT）代码生成的通用逻辑。
       */
      struct ExecutorCodegen {
        // 初始化代码生成器 
        void Init(runtime::Module* m, const Array<Target>& raw_targets) {
          CallFunc("init", m, raw_targets);
          }

        // 核心函数，执行从Relay IR到目标代码的生成
        void Codegen(IRModule mod, const Function& func, String mod_name) {
          CallFunc("codegen", mod, func, mod_name);
          }

        // 纯虚函数，子类必须实现，用于更新最终的BuildOutput
        virtual void UpdateOutput(BuildOutput* ret) = 0;

        // 获取函数元数据
        Map<String, FunctionInfo> GetFunctionMetadata() {
          return CallFunc<Map<String, FunctionInfo>>("get_function_metadata", nullptr);
          }

        // 从代码生成器中提取权重参数
        std::unordered_map<std::string, tvm::runtime::NDArray> GetParams() {
          std::unordered_map<std::string, tvm::runtime::NDArray> ret;
          auto names = CallFunc<Array<runtime::String>>("list_params_name", nullptr);
          for (const auto& expr : names) {
            // Implicit cast from runtime::String to std::string
            std::string key = expr;
            ret[key] = CallFunc<runtime::NDArray>("get_param_by_name", key);
            }
          return ret;
          }

        // 获取外部链接的模块（例如CUDNN, TensorRT等）
        Array<tvm::runtime::Module> GetExternalModules() {
          return CallFunc<Array<tvm::runtime::Module>>("get_external_modules", nullptr);
          }

        // 获取降级（Lowering）后的TIR模块
        Map<Target, IRModule> GetIRModule() {
          return CallFunc<Map<Target, IRModule>>("get_irmodule", nullptr);
          }

        // 获取所有需要的目标设备
        Array<String> ListDevices() { return CallFunc<Array<String>>("get_devices"); }

        // 获取执行器的元数据
        relay::backend::ExecutorCodegenMetadata GetExecutorCodegenMetadata() {
          return CallFunc<relay::backend::ExecutorCodegenMetadata>("get_executor_codegen_metadata");
          }
        virtual ~ExecutorCodegen() {}

        protected:
        // 底层实际的Python侧代码生成器模块
        tvm::runtime::Module mod;

        // 辅助函数，用于调用Python侧模块的PackedFunc
        template <typename R, typename... Args>
        R CallFunc(const std::string& name, Args... args) {
          // 1. 从 mod 对象中，按名字获取一个 PackedFunc
          auto pf = mod.GetFunction(name, false);
          // 2. 调用该 PackedFunc，并传递参数
          return pf(std::forward<Args>(args)...);
          }
        template <typename... Args>
        void CallFunc(const std::string& name, Args... args) {
          auto pf = mod.GetFunction(name, false);
          pf(std::forward<Args>(args)...);
          return;
          }
        };

      /*!
       * \brief AOT (Ahead-of-Time) 执行器的具体代码生成器
       */
      struct AOTCodegen : ExecutorCodegen {
        AOTCodegen() {
          // 从TVM全局函数注册表中获取Python侧的AOT代码生成器
          auto pf = GetPackedFunc("relay.build_module._AOTExecutorCodegen");
          mod = (*pf)();
          }

        // AOT模式不生成graph_json，所以这里设置为空
        void UpdateOutput(BuildOutput* ret) override { ret->graph_json = ""; }

        ~AOTCodegen() {}
        };

      /*!
       * \brief GraphCodegen module wrapper
       Graph Executor 的具体代码生成器
       *
       */
      struct GraphCodegen : ExecutorCodegen {
        GraphCodegen() {
          // 获取Python侧的Graph代码生成器
          auto pf = GetPackedFunc("relay.build_module._GraphExecutorCodegen");
          mod = (*pf)();
          }
        // Graph模式需要更新graph_json
        void UpdateOutput(BuildOutput* ret) override { ret->graph_json = GetGraphJSON(); }

        // 获取图结构的JSON表示
        std::string GetGraphJSON() { return CallFunc<std::string>("get_graph_json", nullptr); }

        ~GraphCodegen() {}
        };


      /*!
       * \brief Executor代码生成器的工厂函数factory function
       * \param executor_str 执行器的名称 ("graph", "aot")
       * \return 一个指向具体代码生成器实例的智能指针
       */
      std::unique_ptr<ExecutorCodegen> MakeExecutorCodegen(String executor_str) {
        std::unique_ptr<ExecutorCodegen> ret;
        if (executor_str == runtime::kTvmExecutorGraph) {
          ret = std::make_unique<GraphCodegen>();
          }
        else if (executor_str == runtime::kTvmExecutorAot) {
          ret = std::make_unique<AOTCodegen>();
          }
        else {
          CHECK(false) << "Executor " << executor_str << " not supported";
          }
        return ret;
        }

      /*!
       * \brief Relay build module
       是对外提供接口的核心类，封装了构建流程的完整逻辑。
       *它被实现为一个 ModuleNode，因此可以被注册到TVM前端（Python）
       */
      class RelayBuildModule : public runtime::ModuleNode {
        public:
        RelayBuildModule() = default;

        /*!
         * \brief Get member function to front-end
         * \param name The name of the function.
         * \param sptr_to_self The pointer to the module node.
         * \return The corresponding member function.
         */
        PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
          // 根据函数名，返回一个打包好的函数(PackedFunc)
          if (name == "get_graph_json") {
            return PackedFunc(
              [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetGraphJSON(); });
            }
          else if (name == "get_module") {
            return PackedFunc(
              [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetModule(); });
            }
          else if (name == "build") {
            return PackedFunc(
              [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
              ICHECK_EQ(args.num_args, 8);
              this->Build(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);
              });
            }
          else if (name == "list_params") {
            return PackedFunc(
              [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->ListParamNames(); });
            }
          else if (name == "get_params") {
            return PackedFunc(
              [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetParams(); });
            }
          else if (name == "set_params") {
            return PackedFunc(
              [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
              Map<String, Constant> params = args[0];
              for (const auto& kv : params) {
                this->SetParam(kv.first, kv.second->data);
                }
              });
            }
          else if (name == "get_devices") {
            return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
              *rv = this->executor_codegen_->ListDevices();
              });
            }
          else if (name == "get_irmodule") {
            return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
              *rv = this->executor_codegen_->GetIRModule();
              });
            }
          else if (name == "get_external_modules") {
            return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
              *rv = this->executor_codegen_->GetExternalModules();
              });
            }
          else if (name == "get_function_metadata") {
            return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
              *rv = this->executor_codegen_->GetFunctionMetadata();
              });
            }
          else if (name == "get_executor_codegen_metadata") {
            return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
              *rv = this->executor_codegen_->GetExecutorCodegenMetadata();
              });
            }
          else if (name == "optimize") {
            return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
              ICHECK_EQ(args.num_args, 2);
              *rv = this->Optimize(args[0], args[1]);
              });
            }
          else {
            LOG(FATAL) << "Unknown packed function: " << name;
            return PackedFunc([sptr_to_self, name](TVMArgs args, TVMRetValue* rv) {});
            }
          }

        /*!
         * \brief Get the GraphJSON for runtime
         *
         * \return const std::string graph_json
         */
        const std::string& GetGraphJSON() { return ret_.graph_json; }

        /*!
         * \brief Get the Module object
         *
         * \return runtime::Module
         */
        runtime::Module GetModule() { return ret_.mod; }

        /*!
         * \brief List all paramter names
         *
         * \return Array<runtime::String> names of params
         */
        Array<runtime::String> ListParamNames() {
          Array<runtime::String> ret;
          for (const auto& kv : params_) {
            ret.push_back(kv.first);
            }
          return ret;
          }

        /*!
         * \brief Get params dictionary
         *
         * \return Map<String, Constant> params dictionary
         */
        Map<String, Constant> GetParams() {
          Map<String, Constant> ret;
          for (const auto& kv : ret_.params) {
            ret.Set(kv.first, Constant(kv.second));
            }
          return ret;
          }

        /*!
         * \brief Set the parameters
         *
         * \param name name of parameter
         * \param data_in input DLTensor
         */
        void SetParam(const std::string& name, runtime::NDArray data_in) { params_[name] = data_in; }

        /*!
         * \brief type key
         *
         * \return const char*
         */
        const char* type_key() const final { return "RelayBuildModule"; }

        /*! \brief Get the property of the runtime module .*/
        int GetPropertyMask() const final { return runtime::ModulePropertyMask::kRunnable; }

        /*!
         * \brief Build relay IRModule for graph executor
         *
         * \param mod Relay IRModule
         * \param raw_targets List of available targets for kernels.
         * \param executor Executor to target
         * \param runtime Runtime to codegen for
         * \param mod_name Name of the module
         */
        void Build(IRModule mod, const Array<Target>& raw_targets, const tvm::Target& target_host,
          const Executor& executor, const Runtime& runtime,
          const WorkspaceMemoryPools& workspace_memory_pools,
          const ConstantMemoryPools& constant_memory_pools, const String mod_name) {
          VLOG_CONTEXT << "Build";
          executor_ = executor;
          runtime_ = runtime;
          workspace_memory_pools_ = workspace_memory_pools;
          constant_memory_pools_ = constant_memory_pools;
          config_ = CompilationConfig(PassContext::Current(), raw_targets);
          VLOG(1) << "Using compilation config:" << std::endl << config_;
          //std::move() 函数的作用是将 mod 变量的状态从左值（lvalue）转换为右值（rvalue），从而允许对其进行移动语义（move semantics）操作。
          BuildRelay(std::move(mod), mod_name);
          }

        protected:
        /*!
         * \brief Optimize a Relay IRModule.
         *
         * \param relay_module The input IRModule where optmization will be applied on.
         * \param raw_targets List of available targets for kernels.
         *
         * \return relay::IRModule The updated Relay IR module after optimization.
         */
        IRModule Optimize(IRModule relay_module, const Array<Target>& raw_targets) {
          VLOG_CONTEXT << "Optimize";
          config_ = CompilationConfig(PassContext::Current(), raw_targets);
          VLOG(1) << "Using compilation config:" << std::endl << config_;
          return OptimizeImpl(std::move(relay_module));
          }

        /*
        *pass优化代码!
        */
        IRModule OptimizeImpl(IRModule relay_module) {
          ICHECK(relay_module.defined()) << "The IRModule must be defined for the Relay compiler.";

          // 1. 将权重参数绑定到模型中
          backend::BindParamsInModule(relay_module, params_);

          // 2. 获取一系列标准的优化Pass
          Array<Pass> pass_seqs =
            GetPassPrefix(/*is_homogenous=*/config_->primitive_targets.size() == 1, /*is_vm=*/false);
          transform::PassContext pass_ctx = PassContext::Current();

          //参数分割（仅同构环境,如果目标设备有最大函数参数限制，则将函数参数拆分以满足这一限制。)
          if (config_->optional_homogeneous_target.defined()) {
            // This pass currently only supports the homogeneous case.
            pass_seqs.push_back(transform::SplitArgs(
              config_->optional_homogeneous_target->GetAttr<Integer>("max_function_args", -1)
              .value()
              .IntValue()));
            }

          // Always plan devices so the remaining passes don't need to distinguish homogeneous vs
          // hetrogenous execution.
          // 3. 规划设备：决定异构计算图中每个部分在哪个设备上运行
          pass_seqs.push_back(transform::PlanDevices(config_));

          // Fuse the operations if it is needed.
          // 4. 算子融合：将多个算子融合成一个，减少开销
          pass_seqs.push_back(transform::FuseOps());

          // Create a sequential pass and perform optimizations.
          // 5. 将所有Pass组合成一个序列并执行
          transform::Pass seq = transform::Sequential(pass_seqs);
          if (config_->optional_homogeneous_target.defined()) {
            With<Target> tctx(config_->optional_homogeneous_target);
            relay_module = seq(relay_module);
            }
          else {
            relay_module = seq(relay_module);
            }

          // Do layout rewrite for auto-scheduler.
          if (backend::IsAutoSchedulerEnabled() && config_->optional_homogeneous_target.defined()) {
            Pass major_pass = transform::AutoSchedulerLayoutRewrite();
            bool enable_layout_rewrite_targets =
              config_->optional_homogeneous_target->GetTargetDeviceType() == kDLCPU ||
              config_->optional_homogeneous_target->GetAttr<String>("device", "") == "mali";
            if (enable_layout_rewrite_targets && pass_ctx.PassEnabled(major_pass->Info())) {
              With<Target> tctx(config_->optional_homogeneous_target);
              relay_module = major_pass(relay_module);
              // Defuse ops to fold constants, then fuse them again
              relay_module = transform::DefuseOps()(relay_module);
              relay_module = transform::FoldConstant()(relay_module);
              relay_module = transform::FuseOps()(relay_module);
              }
            }
          if (backend::IsMetaScheduleEnabled() && config_->optional_homogeneous_target.defined()) {
            Pass major_pass = transform::MetaScheduleLayoutRewrite();
            bool enable_layout_rewrite_targets =
              config_->optional_homogeneous_target->GetTargetDeviceType() == kDLCPU ||
              config_->optional_homogeneous_target->GetAttr<String>("device", "") == "mali";
            if (enable_layout_rewrite_targets && pass_ctx.PassEnabled(major_pass->Info())) {
              With<Target> tctx(config_->optional_homogeneous_target);
              relay_module = major_pass(relay_module);
              // Defuse ops to fold constants, then fuse them again
              relay_module = transform::DefuseOps()(relay_module);
              relay_module = transform::FoldConstant()(relay_module);
              relay_module = transform::FuseOps()(relay_module);
              }
            }

          // ... (一些针对AutoScheduler和MetaSchedule的布局优化Pass)

          // 6. 推断类型，确保图的类型一致性
          relay_module = transform::InferType()(relay_module);

          // Inline the functions that have been lifted by the module scope.
          //
          // TODO(@zhiics) Note that we need to be careful about the subgraphs with
          // global function calls. We should make sure that these callees are also
          // inline functions. However, this should be very unlikely for accelerators
          // and vendor-provided libraries. So we don't handle for now.
          relay_module = transform::Inline()(relay_module);
          relay_module = transform::InferType()(relay_module);
          relay_module = transform::LabelOps()(relay_module);
          relay_module = transform::AnnotateMemoryScope()(relay_module);
          ICHECK(relay_module.defined());

          return relay_module;
          }

        /*!
         * \brief Compile a Relay IR module to runtime module.
         * build 的核心逻辑
         * \param relay_module The Relay IR module.
         * \param params The parameters.
         */
        void BuildRelay(IRModule relay_module, const String& mod_name) {
          printf("ZEdebug:func-BuildRelay-build_module.cc mod_name: %s\n", mod_name.c_str());
          // Relay IRModule -> IRModule optimizations.
          // 步骤1: 调用OptimizeImpl对Relay IRModule进行优化
          // WithAttrs 是一个辅助函数，用于给 IRModule 添加属性
          IRModule module = WithAttrs(
            relay_module, { {tvm::attr::kExecutor, executor_}, {tvm::attr::kRuntime, runtime_} });
          relay_module = OptimizeImpl(std::move(module));

          // Get the updated function and new IRModule to build.
          // Instead of recreating the IRModule, we should look at the differences between this and the
          // incoming IRModule to see if we can just pass (IRModule, Function) to the code generator.
           // 2.从优化后的模块中提取出主函数
          Function func = Downcast<Function>(relay_module->Lookup("main"));
          IRModule func_module = WithAttrs(IRModule::FromExpr(func),
            { {tvm::attr::kExecutor, executor_},
             {tvm::attr::kRuntime, runtime_},
             {tvm::attr::kWorkspaceMemoryPools, workspace_memory_pools_},
             {tvm::attr::kConstantMemoryPools, constant_memory_pools_} });

          // Generate code for the updated function.
          // 3.创建并初始化所选的Executor代码生成器 (AOT or Graph)
          executor_codegen_ = MakeExecutorCodegen(executor_->name);
          executor_codegen_->Init(nullptr, config_->primitive_targets);

          // .执行代码生成。这一步会将Relay IR降级(Lowering)为TensorIR(TIR)！！！！！！！！！！！！！！！！！！！！！！
          executor_codegen_->Codegen(func_module, func, mod_name);

          // .从代码生成器中获取结果
          executor_codegen_->UpdateOutput(&ret_);
          ret_.params = executor_codegen_->GetParams();

          // .获取降级后的TIR模块 tensor IR module
          auto lowered_funcs = executor_codegen_->GetIRModule();//这是一个map?Map<Target, IRModule>?

          // 4.No need to build for external functions.
          Target ext_dev("ext_dev");                  //意思是如果找到一个外部函数就把这个外部函数的IRmod设置为零，因为正常外部已经在外部编译了
          if (lowered_funcs.find(ext_dev) != lowered_funcs.end()) {
            lowered_funcs.Set(ext_dev, IRModule());
            }

          const Target& host_target = config_->host_virtual_device->target;
          const runtime::PackedFunc* pf = runtime::Registry::Get("codegen.LLVMModuleCreate");

          // 5. 调用TIRToRuntime，将TIR编译成目标设备的运行时模块(机器码)
          // When there is no lowered_funcs due to reasons such as optimization.
          if (lowered_funcs.size() == 0) {
            // 如果没有函数需要编译，创建一个空模块
            if (host_target->kind->name == "llvm") {
              CHECK(pf != nullptr) << "Unable to create empty module for llvm without llvm codegen.";
              // 核心编译步骤：TIR -> 机器码
              // If we can decide the target is LLVM, we then create an empty LLVM module.
              ret_.mod = (*pf)(host_target->str(), "empty_module");
              }
            else {
              // If we cannot decide the target is LLVM, we create an empty CSourceModule.
              // The code content is initialized with ";" to prevent complaining
              // from CSourceModuleNode::SaveToFile.
              ret_.mod = tvm::codegen::CSourceModuleCreate(";", "", Array<String>{});
              }
            }
          else {
            ret_.mod = tvm::TIRToRuntime(lowered_funcs, host_target);
            }

          // 6. 创建元数据模块，将所有产物（机器码、权重、外部模块、元数据）打包在一起
          auto ext_mods = executor_codegen_->GetExternalModules();
          ret_.mod = tvm::codegen::CreateMetadataModule(ret_.params, ret_.mod, ext_mods, host_target,
            runtime_, executor_,
            executor_codegen_->GetExecutorCodegenMetadata());
          // Remove external params which were stored in metadata module.
          for (tvm::runtime::Module mod : ext_mods) {
            auto pf_var = mod.GetFunction("get_const_vars");
            if (pf_var != nullptr) {
              Array<String> variables = pf_var();
              for (size_t i = 0; i < variables.size(); i++) {
                auto it = ret_.params.find(variables[i].operator std::string());
                if (it != ret_.params.end()) {
                  VLOG(1) << "constant '" << variables[i] << "' has been captured in external module";
                  ret_.params.erase(it);
                  }
                }
              }
            }
          }

        protected:
        // 指向具体代码生成器（AOT或Graph）的指针
        std::unique_ptr<ExecutorCodegen> executor_codegen_;
        /*! \brief Executor to build for */
        Executor executor_;
        /*! \brief Runtime to codegen for */
        Runtime runtime_;
        /*! \brief Workspace memory pools to codegen for */
        WorkspaceMemoryPools workspace_memory_pools_;
        /*! \brief Constant memory pools to codegen for */
        ConstantMemoryPools constant_memory_pools_;
        /*! \brief parameters */
        std::unordered_map<std::string, runtime::NDArray> params_;
        /*! \brief building output */
        BuildOutput ret_;
        /*! \brief Collects all the targets and scopes we need during compilation. */
        CompilationConfig config_;
        };

      // 创建RelayBuildModule实例的工厂函数 ？？？
      runtime::Module RelayBuildCreate() {
        auto exec = make_object<RelayBuildModule>();// 创建一个 RelayBuildModule 对象
        return runtime::Module(exec);
        }

      // 使用宏将C++实现的RelayBuildModule注册到TVM全局函数表中
      // 这样Python前端就可以通过"relay.build_module._BuildModule"这个名字找到并创建它
      TVM_REGISTER_GLOBAL("relay.build_module._BuildModule").set_body([](TVMArgs args, TVMRetValue* rv) {
        *rv = RelayBuildCreate();
        });

      TVM_REGISTER_GLOBAL("relay.build_module.BindParamsByName")
        .set_body([](TVMArgs args, TVMRetValue* rv) {
        Map<String, Constant> params = args[1];
        std::unordered_map<std::string, runtime::NDArray> params_;
        for (const auto& kv : params) {
          params_[kv.first] = kv.second->data;
          }
        *rv = relay::backend::BindParamsByName(args[0], params_);
          });

      }  // namespace backend
    }  // namespace relay
  }  // namespace tvm
