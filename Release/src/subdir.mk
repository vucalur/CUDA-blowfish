################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/blowfish.cu \
../src/cudaPart.cu 

CU_DEPS += \
./src/blowfish.d \
./src/cudaPart.d 

OBJS += \
./src/blowfish.o \
./src/cudaPart.o 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -O3 -gencode arch=compute_11,code=sm_11 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -O3 -gencode arch=compute_11,code=compute_11 -gencode arch=compute_11,code=sm_11  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


